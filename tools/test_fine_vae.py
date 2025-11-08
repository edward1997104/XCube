import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xcube.data.base import DatasetSpec as DS
from xcube.models.autoencoder import Model as AutoencoderModel
from xcube.utils import exp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reconstruction tests on a single sample using the pretrained fine VAE."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the VAE config YAML (e.g. configs/objaverse/train_vae_128x128x128_sparse.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained VAE checkpoint (e.g. checkpoints/objaverse/fine_vae/last.ckpt).",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        help="Path to a single processed .pkl produced by process_custom_objs.py.",
    )
    parser.add_argument(
        "--samples_root",
        type=Path,
        help="Directory containing per-object folders with mesh.pkl files (e.g. /data/toys4k/512).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./vae_test_results"),
        help="Directory where reconstruction outputs will be stored.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu).",
    )
    parser.add_argument(
        "--use_mode",
        action="store_true",
        help="Use the posterior mean instead of sampling during encoding.",
    )
    parser.add_argument(
        "--extract_mesh",
        action="store_true",
        help="Extract a mesh from the reconstructed grid using the NKSR reconstructor.",
    )
    parser.add_argument(
        "--mesh_config",
        type=Path,
        help="Config YAML for the NKSR reconstructor employed during mesh extraction.",
    )
    parser.add_argument(
        "--mesh_checkpoint",
        type=Path,
        help="Checkpoint for the NKSR reconstructor employed during mesh extraction.",
    )
    parser.add_argument(
        "--mesh_grid_upsample",
        type=int,
        default=2,
        help="Grid upsample factor passed to extract_dual_mesh (default: 2).",
    )
    parser.add_argument(
        "--mesh_max_points",
        type=int,
        default=2_000_000,
        help="Maximum number of evaluation points per chunk during mesh extraction (lower to save VRAM).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Process rank for distributed execution.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="World size for distributed execution (number of parallel ranks).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose outputs already exist in the output directory.",
    )
    return parser.parse_args()


def load_reconstructor(config_path: Path, checkpoint_path: Path, device: torch.device):
    if not config_path or not checkpoint_path:
        raise ValueError("Both mesh config and checkpoint paths must be provided.")
    if not config_path.exists():
        raise FileNotFoundError(f"Mesh config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Mesh checkpoint not found: {checkpoint_path}")

    hparams = exp.parse_config_yaml(config_path)
    if not hasattr(hparams, "model"):
        raise ValueError("Mesh config does not declare a `model` entry.")

    module = importlib.import_module(f"xcube.models.{hparams.model}").Model
    reconstructor = module.load_from_checkpoint(checkpoint_path, hparams=hparams, strict=False)
    reconstructor = reconstructor.to(device)
    reconstructor.eval()
    return reconstructor


def extract_mesh_from_prediction(
    reconstructor,
    decoded_grid,
    normal_field,
    grid_upsample: int,
    max_points: int,
):
    if decoded_grid.grid_count == 0:
        raise RuntimeError("Decoded grid is empty; cannot extract a mesh.")
    if normal_field is None:
        raise RuntimeError(
            "The VAE was trained without a normal branch, so mesh extraction is unavailable."
        )

    pd_grid = decoded_grid[0]
    pd_normal = normal_field.feature[0].jdata

    with torch.no_grad():
        outputs = reconstructor.forward({"in_grid": pd_grid, "in_normal": pd_normal})

    mesh_field = None
    selected_key = None
    for key in ("kernel_sdf", "neural_udf"):
        if key in outputs:
            mesh_field = outputs[key]
            selected_key = key
            break

    if mesh_field is None:
        available = ", ".join(outputs.keys())
        raise RuntimeError(
            "Reconstructor output does not contain a mesh-compatible field. "
            f"Available keys: {available}"
        )

    mesh = mesh_field.extract_dual_mesh(grid_upsample=grid_upsample, max_points=max_points)
    return mesh, selected_key


def save_mesh_as_obj(mesh, destination: Path):
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Saving meshes requires the `trimesh` package. Please install it in the active environment."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh_obj = trimesh.Trimesh(
        vertices=mesh.v.detach().cpu().numpy(),
        faces=mesh.f.detach().cpu().numpy(),
    )
    mesh_obj.export(destination)
    return destination


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def load_sample(sample_path: Path, device: torch.device):
    data = torch.load(sample_path, map_location="cpu")
    if "points" not in data or "normals" not in data:
        raise ValueError(f"{sample_path} does not contain the expected keys (points/normals).")

    grid = data["points"].to(device)
    normals = data["normals"].jdata.to(device)

    batch = {
        DS.INPUT_PC: grid,
        DS.TARGET_NORMAL: [normals],
    }
    meta = data.get("meta", {})
    return batch, meta


def run_reconstruction(vae: AutoencoderModel, batch, use_mode: bool):
    with torch.no_grad():
        latents = vae._encode(batch, use_mode=use_mode)
        features = vae.unet.FeaturesSet()
        features, decoded = vae.unet.decode(features, latents, is_testing=True)

    input_grid = batch[DS.INPUT_PC]
    input_xyz = (
        input_grid.grid_to_world(input_grid.ijk.float()).jdata.detach().cpu().numpy()
    )
    pred_grid = features.structure_grid[0]
    pred_xyz = (
        pred_grid.grid_to_world(pred_grid.ijk.float()).jdata.detach().cpu().numpy()
    )

    info = {
        "latent_voxels": int(latents.grid.total_voxels),
        "input_voxels": int(input_grid.total_voxels),
        "recon_voxels": int(pred_grid.total_voxels),
    }
    return {
        "input_xyz": input_xyz,
        "pred_xyz": pred_xyz,
        "info": info,
        "decoded_grid": decoded.grid,
        "normal_field": features.normal_features.get(-1),
    }


def derive_sample_label(sample_path: Path) -> str:
    parent = sample_path.parent.name
    if parent:
        return parent
    return sample_path.stem


def collect_samples(args) -> list[Path]:
    if args.samples_root:
        if not args.samples_root.exists():
            raise FileNotFoundError(f"Samples root not found: {args.samples_root}")
        sample_paths = sorted(
            args.samples_root.glob("*/mesh.pkl"),
            key=lambda p: (p.parent.name, p.name),
        )
        if not sample_paths:
            raise SystemExit(f"No mesh.pkl files found under {args.samples_root}")
    elif args.sample:
        if not args.sample.exists():
            raise FileNotFoundError(f"Sample file not found: {args.sample}")
        sample_paths = [args.sample]
    else:
        raise SystemExit("Please provide either --sample or --samples_root.")

    if args.world_size < 1:
        raise ValueError("--world_size must be >= 1.")
    if not (0 <= args.rank < args.world_size):
        raise ValueError("--rank must be in [0, world_size).")

    if args.world_size == 1:
        return sample_paths
    return [p for idx, p in enumerate(sample_paths) if idx % args.world_size == args.rank]


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    hparams = exp.parse_config_yaml(args.config)
    vae = AutoencoderModel.load_from_checkpoint(args.checkpoint, hparams=hparams)
    vae = vae.to(device)
    vae.eval()

    sample_paths = collect_samples(args)
    if not sample_paths:
        print(f"No samples assigned to rank {args.rank}.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reconstructor = None
    errors = []
    for sample_path in sample_paths:
        label = derive_sample_label(sample_path)
        npz_path = args.output_dir / f"{label}_recon.npz"
        info_path = args.output_dir / f"{label}_info.json"
        mesh_path = args.output_dir / f"{label}.obj" if args.extract_mesh else None

        if (
            args.skip_existing
            and npz_path.exists()
            and info_path.exists()
            and (not mesh_path or mesh_path.exists())
        ):
            print(f"[SKIP] {label} (outputs already exist)")
            continue

        batch = None
        recon = None
        mesh = None
        try:
            batch, sample_meta = load_sample(sample_path, device)
            recon = run_reconstruction(vae, batch, args.use_mode)

            np.savez(
                npz_path,
                input_points=recon["input_xyz"],
                recon_points=recon["pred_xyz"],
            )

            mesh_field_key = None
            if args.extract_mesh:
                if args.mesh_config is None or args.mesh_checkpoint is None:
                    raise ValueError("--mesh_config and --mesh_checkpoint are required when --extract_mesh is set.")
                if reconstructor is None:
                    reconstructor = load_reconstructor(args.mesh_config, args.mesh_checkpoint, device)
                mesh, mesh_field_key = extract_mesh_from_prediction(
                    reconstructor,
                    recon["decoded_grid"],
                    recon["normal_field"],
                    args.mesh_grid_upsample,
                    args.mesh_max_points,
                )
                save_mesh_as_obj(mesh, mesh_path)

            with info_path.open("w", encoding="utf-8") as f:
                summary = {
                    "sample": str(sample_path),
                    "sample_label": label,
                    "config": str(args.config),
                    "checkpoint": str(args.checkpoint),
                    "use_mode": args.use_mode,
                    "rank": args.rank,
                    "world_size": args.world_size,
                }
                summary.update(recon["info"])
                if sample_meta:
                    summary["sample_meta"] = sample_meta
                if mesh_path is not None:
                    summary["mesh"] = str(mesh_path)
                    summary["mesh_field"] = mesh_field_key
                    summary["mesh_grid_upsample"] = args.mesh_grid_upsample
                    summary["mesh_max_points"] = args.mesh_max_points
                json.dump(summary, f, indent=2)

            print(f"[OK] {label} -> {npz_path}")
        except Exception as exc:  # pylint: disable=broad-except
            errors.append((sample_path, str(exc)))
            print(f"[ERR] Failed on {sample_path}: {exc}")
        finally:
            if batch is not None:
                del batch
            if recon is not None:
                del recon
            if mesh is not None:
                del mesh
            torch.cuda.empty_cache()

    if errors:
        print(f"\nCompleted with {len(errors)} failures:")
        for sample_path, err in errors[:10]:
            print(f" - {sample_path}: {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more.")
    else:
        print("\nAll samples processed successfully.")


if __name__ == "__main__":
    main()
