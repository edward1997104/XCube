
import os
import point_cloud_utils as pcu
import numpy as np
from tqdm import tqdm
import fvdb
import torch
import argparse
import multiprocessing as mp

args = argparse.ArgumentParser()
args.add_argument('--data_root', type=str, default='../data/shapenet_manifold')
args.add_argument('--target_root', type=str, default='../data/shapenet/')
args.add_argument('--num_vox', type=int, default=512)
args.add_argument('--categories', type=str, default='03001627')
args.add_argument('--num_split', type=int, default=8)
args.add_argument('--split_id', type=int, default=0)
args.add_argument('--workers', type=int, default=8, help='Number of parallel worker processes.')
args.add_argument('--device', type=str, default='cuda', help='Device for FVDB/Torch ops (cuda or cpu).')
args = args.parse_args()

data_root = args.data_root
target_root = args.target_root

_shapenet_categories = args.categories.split(',')
num_vox = args.num_vox

if num_vox > 512:
    max_num_vox = num_vox
    sample_pcs_num = 5_000_000
else:
    max_num_vox = 512
    sample_pcs_num = 1_000_000
vox_size = 1.0 / max_num_vox

_SN_WORKER_CFG = {}


def _init_sn_worker(config):
    global _SN_WORKER_CFG
    _SN_WORKER_CFG = config


def _process_shapenet_model(model_id):
    cfg = _SN_WORKER_CFG
    category_dir = cfg["category_dir"]
    target_dir = cfg["target_dir"]
    vox_size = cfg["vox_size"]
    num_vox = cfg["num_vox"]
    sample_pcs_num = cfg["sample_pcs_num"]
    device = cfg["device"]

    target_path = target_dir / f"{model_id.split('-')[0]}.pkl"
    if target_path.exists():
        return True

    model_path = category_dir / model_id
    try:
        v, f = pcu.load_mesh_vf(str(model_path))

        faces_int = f.astype(np.int32)
        try:
            fid, bc = pcu.sample_mesh_random(v, faces_int, sample_pcs_num)
            ref_xyz = pcu.interpolate_barycentric_coords(faces_int, fid, bc, v)
        except Exception:  # pylint: disable=broad-except
            fid, bc = pcu.sample_mesh_random(v, faces_int, sample_pcs_num)
            ref_xyz = pcu.interpolate_barycentric_coords(faces_int, fid, bc, v)

        n = pcu.estimate_mesh_face_normals(v, faces_int)
        ref_normal = n[fid]

        ijk = pcu.voxelize_triangle_mesh(v, faces_int, vox_size, np.zeros(3))
        ijk_tensor = torch.from_numpy(ijk).to(device)
        grid = fvdb.sparse_grid_from_ijk(
            fvdb.JaggedTensor([ijk_tensor]), voxel_sizes=vox_size, origins=[vox_size / 2.0] * 3
        )

        ref_xyz = torch.from_numpy(ref_xyz).float().to(device)
        ref_normal = torch.from_numpy(ref_normal).float().to(device)
        input_normal = grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz), fvdb.JaggedTensor(ref_normal))
        input_normal.jdata /= (input_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)

        xyz = grid.grid_to_world(grid.ijk.float()).jdata
        xyz_norm = xyz * 128 / 100
        ref_xyz_scaled = ref_xyz * 128 / 100

        if num_vox == 512:
            target_voxel_size = 0.001953125
            target_grid = fvdb.sparse_grid_from_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
            )
        elif num_vox == 16:
            target_voxel_size = 0.08
            target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
            )
        elif num_vox == 128:
            target_voxel_size = 0.01
            target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
            )
        elif num_vox == 256:
            target_voxel_size = 0.005
            target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
            )
        elif num_vox == 1024:
            target_voxel_size = 0.00125
            target_grid = fvdb.sparse_grid_from_points(
                fvdb.JaggedTensor(xyz_norm), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.0] * 3
            )
        else:
            raise NotImplementedError

        target_normal = target_grid.splat_trilinear(fvdb.JaggedTensor(ref_xyz_scaled), fvdb.JaggedTensor(ref_normal))
        target_normal.jdata /= (target_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)

        save_dict = {
            "points": target_grid.to("cpu"),
            "normals": target_normal.cpu(),
            "ref_xyz": ref_xyz_scaled.cpu(),
            "ref_normal": ref_normal.cpu(),
        }

        torch.save(save_dict, target_path)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        return False, model_id, str(exc)


for category in _shapenet_categories:
    category_dir = Path(data_root) / category
    print(category_dir)
    model_ids = sorted([f for f in os.listdir(category_dir) if os.path.isfile(category_dir / f) and f.endswith('.ply')])
    num_models = len(model_ids)
    print(num_models)
    num_models_per_split = num_models // args.num_split
    if args.split_id == args.num_split - 1:
        model_ids = model_ids[args.split_id * num_models_per_split:]
    else:
        model_ids = model_ids[args.split_id * num_models_per_split: (args.split_id + 1) * num_models_per_split]
    
    print(f"Processing {len(model_ids)} models in split {args.split_id} of category {category}")
    target_dir = Path(target_root) / f"{num_vox}" / category
    target_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "category_dir": category_dir,
        "target_dir": target_dir,
        "vox_size": vox_size,
        "num_vox": num_vox,
        "sample_pcs_num": sample_pcs_num,
        "device": args.device,
    }
    ctx = mp.get_context("spawn")
    errors = []
    with ctx.Pool(processes=args.workers, initializer=_init_sn_worker, initargs=(config,)) as pool:
        for result in tqdm(pool.imap_unordered(_process_shapenet_model, model_ids), total=len(model_ids), desc=f"{category}"):
            if isinstance(result, tuple) and not result[0]:
                errors.append((category, result[1], result[2]))

    if errors:
        print(f"[WARN] {len(errors)} failures encountered for category {category}:")
        for cat, mid, err in errors[:10]:
            print(f" - {cat}/{mid}: {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more.")
