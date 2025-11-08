import argparse
import json
from pathlib import Path

import multiprocessing as mp

import fvdb
import numpy as np
import point_cloud_utils as pcu
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert OBJ meshes into FVDB sparse grids using the exact ShapeNet preprocessing recipe."
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
        help="Target voxel resolution (matches the ShapeNet script's num_vox argument).",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes.",
    )
    return parser.parse_args()


def ensure_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def determine_sampling_params(resolution: int):
    """Replicates the sample_pcs_num / voxel size logic from datagen/shapenet_example.py."""
    if resolution > 512:
        max_num_vox = resolution
        sample_pcs_num = 5_000_000
    else:
        max_num_vox = 512
        sample_pcs_num = 1_000_000
    voxel_size = 1.0 / float(max_num_vox)
    return voxel_size, sample_pcs_num


def voxelise_with_shapenet_scheme(vertices: np.ndarray, faces: np.ndarray, voxel_size: float, device: torch.device):
    faces_int = faces.astype(np.int32, copy=False)
    origin = np.zeros(3, dtype=np.float32)
    ijk = pcu.voxelize_triangle_mesh(vertices, faces_int, voxel_size, origin)
    if ijk.size == 0:
        raise RuntimeError("Voxelisation produced an empty grid.")
    ijk_tensor = torch.from_numpy(ijk.astype(np.int32)).to(device)
    grid = fvdb.sparse_grid_from_ijk(
        fvdb.JaggedTensor([ijk_tensor]),
        voxel_sizes=voxel_size,
        origins=[voxel_size / 2.0] * 3,
    )
    return grid


def sample_reference_points(vertices: np.ndarray, faces: np.ndarray, num_samples: int):
    faces_int = faces.astype(np.int32, copy=False)
    try:
        fid, bc = pcu.sample_mesh_random(vertices, faces_int, num_samples)
    except RuntimeError:
        fid, bc = pcu.sample_mesh(vertices, faces_int, num_samples)
    ref_xyz = pcu.interpolate_barycentric_coords(faces_int, fid, bc, vertices)
    face_normals = pcu.estimate_mesh_face_normals(vertices, faces_int)
    ref_normal = face_normals[fid]
    return ref_xyz.astype(np.float32), ref_normal.astype(np.float32)


def scale_and_build_target_grid(grid: fvdb.GridBatch, xyz_scale: float, resolution: int):
    """Creates the target FVDB grid exactly like the ShapeNet example."""
    world_xyz = grid.grid_to_world(grid.ijk.float()).jdata
    xyz_norm = fvdb.JaggedTensor([world_xyz * xyz_scale])

    if resolution == 512:
        target_voxel_size = 0.001953125
        target_grid = fvdb.sparse_grid_from_points(
            xyz_norm,
            voxel_sizes=target_voxel_size,
            origins=[target_voxel_size / 2.0] * 3,
        )
    elif resolution == 16:
        target_voxel_size = 0.08
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
            xyz_norm, voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
        )
    elif resolution == 128:
        target_voxel_size = 0.01
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
            xyz_norm, voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
        )
    elif resolution == 256:
        target_voxel_size = 0.005
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
            xyz_norm, voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
        )
    elif resolution == 1024:
        target_voxel_size = 0.00125
        target_grid = fvdb.sparse_grid_from_points(
            xyz_norm,
            voxel_sizes=target_voxel_size,
            origins=[target_voxel_size / 2.0] * 3,
        )
    else:
        raise NotImplementedError(f"Resolution {resolution} is not supported by the ShapeNet preprocessing recipe.")
    return target_grid


def process_mesh(obj_path: Path, resolution: int, device: torch.device):
    vertices, faces = pcu.load_mesh_vf(str(obj_path))
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("Empty mesh or failed to load faces.")

    base_voxel_size, sample_pcs_num = determine_sampling_params(resolution)
    grid = voxelise_with_shapenet_scheme(vertices, faces, base_voxel_size, device)

    ref_xyz, ref_normal = sample_reference_points(vertices, faces, sample_pcs_num)
    ref_xyz_tensor = torch.from_numpy(ref_xyz).to(device)
    ref_normal_tensor = torch.from_numpy(ref_normal).to(device)

    # Splat normals just like shapenet_example.py
    splatted = grid.splat_trilinear(
        fvdb.JaggedTensor([ref_xyz_tensor]),
        fvdb.JaggedTensor([ref_normal_tensor]),
    )
    splatted_norm = splatted.jdata.norm(dim=1, keepdim=True).clamp_min_(1e-6)
    splatted.jdata /= splatted_norm

    scale = 128.0 / 100.0
    target_grid = scale_and_build_target_grid(grid, scale, resolution)

    ref_xyz_scaled = ref_xyz_tensor * scale
    target_normal = target_grid.splat_trilinear(
        fvdb.JaggedTensor([ref_xyz_scaled]),
        fvdb.JaggedTensor([ref_normal_tensor]),
    )
    target_norm = target_normal.jdata.norm(dim=1, keepdim=True).clamp_min_(1e-6)
    target_normal.jdata /= target_norm

    processed = {
        "points": target_grid.to("cpu"),
        "normals": target_normal.cpu(),
        "ref_xyz": ref_xyz_scaled.cpu(),
        "ref_normal": ref_normal_tensor.cpu(),
        "meta": {
            "source_obj": str(obj_path),
            "base_voxel_size": base_voxel_size,
            "resolution": resolution,
            "sample_points": sample_pcs_num,
        },
    }
    return processed


_WORKER_CFG = {}
_WORKER_DEVICE = None


def _init_worker(config: dict):
    global _WORKER_CFG, _WORKER_DEVICE
    _WORKER_CFG = config
    _WORKER_DEVICE = ensure_device(config["device"])


def _process_task(task):
    obj_path_str, category = task
    cfg = _WORKER_CFG
    obj_path = Path(obj_path_str)
    target_dir = cfg["output_root"] / str(cfg["resolution"]) / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{obj_path.stem}.pkl"

    if target_path.exists() and not cfg["overwrite"]:
        return True

    try:
        processed = process_mesh(obj_path, cfg["resolution"], _WORKER_DEVICE)
    except Exception as exc:  # pylint: disable=broad-except
        return False, obj_path_str, str(exc)

    torch.save(processed, target_path)
    return True


def main():
    args = parse_args()
    obj_files = sorted(args.input_dir.rglob("*.obj"))
    if not obj_files:
        raise SystemExit(f"No OBJ files found under {args.input_dir}")

    output_root = args.output_root
    (output_root / str(args.resolution)).mkdir(parents=True, exist_ok=True)

    tasks = []
    for obj_path in obj_files:
        if args.category_mode == "parent":
            category = obj_path.parent.name
        else:
            category = args.category
        tasks.append((str(obj_path), category))

    if not tasks:
        print("No meshes to process.")
        return

    worker_cfg = {
        "resolution": args.resolution,
        "device": args.device,
        "output_root": output_root,
        "overwrite": args.overwrite,
    }
    ctx = mp.get_context("spawn")
    errors = []
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_cfg,)) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_task, tasks),
            total=len(tasks),
            desc="Processing meshes",
        ):
            if isinstance(result, tuple) and not result[0]:
                errors.append(result)

    if errors:
        print(f"[WARN] {len(errors)} meshes failed during processing:")
        for _, mesh_path, err in errors[:10]:
            print(f" - {mesh_path}: {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more.")

    # Summarise processed dataset for convenience.
    _, summary_samples = determine_sampling_params(args.resolution)
    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(output_root),
        "resolution": args.resolution,
        "num_samples": summary_samples,
        "device": args.device,
        "workers": args.workers,
    }
    summary_path = output_root / str(args.resolution) / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Processing complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
