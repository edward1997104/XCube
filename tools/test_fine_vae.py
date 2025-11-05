import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
        required=True,
        help="Path to the processed .pkl produced by process_custom_objs.py.",
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
    return parser.parse_args()


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
    return input_xyz, pred_xyz, info


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.sample.exists():
        raise FileNotFoundError(f"Sample file not found: {args.sample}")

    hparams = exp.parse_config_yaml(args.config)
    vae = AutoencoderModel.load_from_checkpoint(args.checkpoint, hparams=hparams)
    vae = vae.to(device)
    vae.eval()

    batch, sample_meta = load_sample(args.sample, device)
    input_xyz, pred_xyz, info = run_reconstruction(vae, batch, args.use_mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = args.sample.stem
    np.savez(
        args.output_dir / f"{sample_name}_recon.npz",
        input_points=input_xyz,
        recon_points=pred_xyz,
    )
    with (args.output_dir / f"{sample_name}_info.json").open("w", encoding="utf-8") as f:
        summary = {
            "sample": str(args.sample),
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "use_mode": args.use_mode,
        }
        summary.update(info)
        if sample_meta:
            summary["sample_meta"] = sample_meta
        json.dump(summary, f, indent=2)

    print("Reconstruction complete:")
    print(json.dumps(info, indent=2))
    print(f"Outputs saved under {args.output_dir}")


if __name__ == "__main__":
    main()
