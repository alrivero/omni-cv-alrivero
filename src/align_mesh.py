"""
=====================================================================
 * MIT License
 *
 * Copyright (c) 2025 Omni Instrument Inc.
 * ...
 * =====================================================================
"""

from __future__ import annotations
import numpy as np
import open3d as o3d
import argparse
import os


def align_to_gt(recon_path: str, gt_path: str, output_path: str,
                voxel_size: float = 0.1) -> o3d.pipelines.registration.RegistrationResult:

    recon_mesh = o3d.io.read_triangle_mesh(recon_path)
    gt_mesh    = o3d.io.read_triangle_mesh(gt_path)

    recon_pcd = recon_mesh.sample_points_uniformly(number_of_points=200_000)
    gt_pcd    = gt_mesh.sample_points_uniformly(number_of_points=200_000)

    recon_down = recon_pcd.voxel_down_sample(voxel_size)
    gt_down    = gt_pcd.voxel_down_sample(voxel_size)

    recon_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    gt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Coarse init: translate reconstruction centre of mass onto GT centre of mass
    t_init = np.eye(4)
    t_init[:3, 3] = gt_down.get_center() - recon_down.get_center()

    # Fine ICP (point-to-plane)
    result = o3d.pipelines.registration.registration_icp(
        recon_down, gt_down,
        max_correspondence_distance=voxel_size * 5,
        init=t_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100)
    )

    recon_mesh.transform(result.transformation)
    o3d.io.write_triangle_mesh(output_path, recon_mesh)

    return result


if __name__ == "__main__":
    home = os.path.expanduser("~")
    default_gt    = os.path.join(home, "dataset", "meshes", "omni_mesh.stl")
    default_recon = os.path.join(home, "output", "mesh.stl")
    default_out   = os.path.join(os.path.dirname(default_recon), "mesh_aligned.stl")

    parser = argparse.ArgumentParser("ICP alignment: reconstruction → ground truth")
    parser.add_argument("--gt",     default=default_gt)
    parser.add_argument("--recon",  default=default_recon)
    parser.add_argument("--output", default=default_out,
                        help="Where to save the aligned reconstruction mesh")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size for ICP downsampling (metres, default 0.1)")
    args = parser.parse_args()

    print(f"GT:    {args.gt}")
    print(f"Recon: {args.recon}")
    print(f"Aligning...")

    result = align_to_gt(args.recon, args.gt, args.output, args.voxel_size)

    print(f"\n===== ICP Result =====")
    print(f"Fitness (overlap):  {result.fitness:.4f}   (1.0 = perfect overlap)")
    print(f"Inlier RMSE:        {result.inlier_rmse:.4f} m")
    print(f"Aligned mesh saved: {args.output}")
    print("======================\n")
    print("Now run compute_metrics.py with --recon pointing at the aligned mesh:")
    print(f"  python3 src/compute_metrics.py --gt {args.gt} --recon {args.output}")
