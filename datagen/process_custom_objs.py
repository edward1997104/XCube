import argparse
import json
from pathlib import Path

import fvdb
import numpy as np
import point_cloud_utils as pcu
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OBJ meshes into FVDB sparse grids for VAE testing."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory that contains OBJ meshes (recursively processed).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory where processed .pkl files will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Target voxel resolution (use 512 for the fine VAE).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1_000_000,
        help="Number of random surface samples for normal splatting.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run FVDB ops on (cuda or cpu). Defaults to cuda when available.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="custom",
        help="Category name used to organise outputs (folder under output_root/resolution).",
    )
    parser.add_argument(
        "--category_mode",
        choices=["fixed", "parent"],
        default="fixed",
        help="Use a fixed category name or derive it from each mesh's parent directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if the target .pkl already exists.",
    )
    return parser.parse_args()


def ensure_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def normalise_vertices(vertices: np.ndarray):
    """Centre and scale vertices to fit inside a 2x2x2 cube, returning shifted coords and transform info."""
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    center = (min_corner + max_corner) * 0.5
    half_extent = (max_corner - min_corner).max() * 0.5
    half_extent = float(max(half_extent, 1e-6))
    centred = vertices - center
    scaled = centred / half_extent  # in [-1, 1]
    shifted = scaled + 1.0  # shift to [0, 2] for voxelisation with positive coordinates
    transform = {
        "center": center.tolist(),
        "half_extent": half_extent,
    }
    return shifted.astype(np.float32), transform


def voxelise_mesh(vertices: np.ndarray, faces: np.ndarray, resolution: int, device: torch.device):
    voxel_size = 2.0 / float(resolution)
    origin = np.zeros(3, dtype=np.float32)
    ijk = pcu.voxelize_triangle_mesh(vertices, faces.astype(np.int32), voxel_size, origin)
    if ijk.size == 0:
        raise RuntimeError("Voxelisation produced an empty grid.")
    ijk_tensor = torch.from_numpy(ijk.astype(np.int32)).to(device)
    grid = fvdb.sparse_grid_from_ijk(
        fvdb.JaggedTensor([ijk_tensor]),
        voxel_sizes=voxel_size,
        origins=[voxel_size / 2.0] * 3,
    )
    return grid, voxel_size


def splat_normals(grid: fvdb.GridBatch, vertices: np.ndarray, faces: np.ndarray, num_samples: int, device: torch.device):
    face_normals = pcu.estimate_mesh_face_normals(vertices, faces.astype(np.int32))
    # Sample uniformly over triangles; fall back to deterministic sampling if random fails.
    try:
        fid, bc = pcu.sample_mesh_random(vertices, faces.astype(np.int32), num_samples)
    except RuntimeError:
        # Degenerate mesh – use evenly spaced sampling as a fallback.
        fid, bc = pcu.sample_mesh(vertices, faces.astype(np.int32), num_samples)
    ref_xyz = pcu.interpolate_barycentric_coords(faces.astype(np.int32), fid, bc, vertices)
    ref_normal = face_normals[fid]

    ref_xyz_tensor = torch.from_numpy(ref_xyz.astype(np.float32)).to(device)
    ref_normal_tensor = torch.from_numpy(ref_normal.astype(np.float32)).to(device)

    jagged_xyz = fvdb.JaggedTensor([ref_xyz_tensor])
    jagged_normal = fvdb.JaggedTensor([ref_normal_tensor])
    splatted = grid.splat_trilinear(jagged_xyz, jagged_normal)
    # Normalise per-voxel normals.
    norms = splatted.jdata.norm(dim=1, keepdim=True).clamp_min_(1e-6)
    splatted.jdata /= norms
    return splatted


def process_mesh(obj_path: Path, args, device: torch.device):
    vertices, faces = pcu.load_mesh_vf(str(obj_path))
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("Empty mesh or failed to load faces.")
    norm_vertices, transform = normalise_vertices(vertices)
    grid, voxel_size = voxelise_mesh(norm_vertices, faces, args.resolution, device)
    normals = splat_normals(grid, norm_vertices, faces, args.num_samples, device)

    # Assemble output dictionary; store transform for potential inverse mapping.
    processed = {
        "points": grid.to("cpu"),
        "normals": normals.cpu(),
        "meta": {
            "source_obj": str(obj_path),
            "voxel_size": voxel_size,
            "transform": transform,
            "resolution": args.resolution,
        },
    }
    return processed


def main():
    args = parse_args()
    device = ensure_device(args.device)

    obj_files = sorted(args.input_dir.rglob("*.obj"))
    if not obj_files:
        raise SystemExit(f"No OBJ files found under {args.input_dir}")

    output_root = args.output_root
    (output_root / str(args.resolution)).mkdir(parents=True, exist_ok=True)

    for obj_path in tqdm(obj_files, desc="Processing meshes"):
        if args.category_mode == "parent":
            category = obj_path.parent.name
        else:
            category = args.category

        target_dir = output_root / str(args.resolution) / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{obj_path.stem}.pkl"

        if target_path.exists() and not args.overwrite:
            continue

        try:
            processed = process_mesh(obj_path, args, device)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to process {obj_path}: {exc}")
            continue

        torch.save(processed, target_path)

    # Summarise processed dataset for convenience.
    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(output_root),
        "resolution": args.resolution,
        "num_samples": args.num_samples,
        "device": str(device),
    }
    summary_path = output_root / str(args.resolution) / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Processing complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
