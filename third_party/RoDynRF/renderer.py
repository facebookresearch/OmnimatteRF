# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from dataLoader.ray_utils import (get_ray_directions_blender, get_rays,
                                  get_rays_with_batch, ndc_rays_blender)
from flow_viz import flow_to_image
from PIL import Image
from tqdm.auto import tqdm

from utils import (rgb_lpips, rgb_ssim, visualize_array, visualize_normalized_depths, save_image_np)


def OctreeRender_trilinear_fast(rays, ts, timeembeddings, tensorf, xyz_sampled, z_vals_input, ray_valid, chunk=4096, N_samples=-1, ray_type="ndc", white_bg=True, is_train=False, device='cuda'):

    rgbs, depth_maps, blending_maps, pts_refs, weights_ds, delta_xyzs, rgb_points, sigmas, z_vals, dists = [
    ], [], [], [], [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        ts_chunk = ts[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        # allposes_refine_train_chunk = allposes_refine_train[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        xyz_sampled_chunk = xyz_sampled[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        z_vals_chunk = z_vals_input[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        ray_valid_chunk = ray_valid[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        timeembeddings_chunk = None
        if timeembeddings is not None:
            timeembeddings_chunk = timeembeddings[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        # if is_train:
        rgb_map, depth_map, blending_map, pts_ref, weights_d, xyz_prime, rgb_point, sigma, z_val, dist = tensorf(
            rays_chunk, ts_chunk, timeembeddings_chunk, xyz_sampled_chunk, z_vals_chunk, ray_valid_chunk, is_train=is_train, white_bg=white_bg, ray_type=ray_type, N_samples=N_samples)
        delta_xyz = xyz_prime - xyz_sampled_chunk

        # rgb_map, depth_map, blending_map, pts_ref, weights_d, delta_xyz, rgb_point, sigma, z_val, dist = tensorf(rays_chunk, ts_chunk, timeembeddings_chunk, None, None, None, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        # else:
        #     rgb_map, depth_map = tensorf(rays_chunk, ts_chunk, timeembeddings_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        if blending_map is None:
            # if not is_train:
            #     weights_d = weights_d.cpu()
            #     rgb_point = rgb_point.cpu()
            #     sigma = sigma.cpu()
            #     z_val = z_val.cpu()
            #     dist = dist.cpu()
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
            pts_refs.append(pts_ref)
            weights_ds.append(weights_d)
            rgb_points.append(rgb_point)
            sigmas.append(sigma)
            z_vals.append(z_val)
            dists.append(dist)
            continue
        # scene_flow_f, scene_flow_b = tensorf.module.get_forward_backward_scene_flow(pts_ref, ts_chunk.to(device))
        # pts_ref_f, pts_ref_b, scene_flow_f, scene_flow_b = tensorf.get_forward_backward_scene_flow_point(pts_ref, ts_chunk.to(device), weights_d, rays_chunk)

        # scene_flow_f = (scene_flow_f / (torch.abs(scene_flow_f).max()) + 1.0)/2.0

        # if not is_train:

        #     weights_d = weights_d.cpu()
        #     rgb_point = rgb_point.cpu()
        #     sigma = sigma.cpu()
        #     z_val = z_val.cpu()
        #     dist = dist.cpu()
        #     pts_ref = pts_ref.cpu()
        #     # pts_ref_f = pts_ref_f.cpu()
        #     # pts_ref_b = pts_ref_b.cpu()
        #     sceneflow_f = sceneflow_f.cpu()
        #     sceneflow_b = sceneflow_b.cpu()
        #     weights_d = weights_d.cpu()
        #     delta_xyz = delta_xyz.cpu()
        #     # scene_flow_f = scene_flow_f.cpu()
        #     rgb_point = rgb_point.cpu()
        #     sigma = sigma.cpu()
        #     z_val = z_val.cpu()
            # dist = dist.cpu()

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        blending_maps.append(blending_map)
        pts_refs.append(pts_ref)
        # pts_fs.append(pts_ref_f)
        # pts_bs.append(pts_ref_b)
        weights_ds.append(weights_d)
        delta_xyzs.append(delta_xyz)
        # scene_flow_maps.append(scene_flow_f)
        rgb_points.append(rgb_point)
        sigmas.append(sigma)
        z_vals.append(z_val)
        dists.append(dist)
    # if is_train:
    if len(blending_maps) == 0:
        return torch.cat(rgbs), torch.cat(depth_maps), None, torch.cat(pts_refs), torch.cat(weights_ds), None, None, torch.cat(rgb_points), torch.cat(sigmas), torch.cat(z_vals), torch.cat(dists)
    else:
        return torch.cat(rgbs), torch.cat(depth_maps), torch.cat(blending_maps), torch.cat(pts_refs), torch.cat(weights_ds), torch.cat(delta_xyzs), None, torch.cat(rgb_points), torch.cat(sigmas), torch.cat(z_vals), torch.cat(dists)
# else:
#     return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def sampleXYZ(tensorf, rays_train, N_samples, ray_type="ndc", is_train=False, multiGPU=False):
    # fix random sample for static and dynamic
    if multiGPU:
        if ray_type == "ndc":
            xyz_sampled, z_vals, ray_valid = tensorf.module.sample_ray_ndc(
                rays_train[:, :3], rays_train[:, 3:6], is_train=is_train, N_samples=N_samples)
        elif ray_type == "contract":
            xyz_sampled, z_vals, ray_valid = tensorf.module.sample_ray_contracted(
                rays_train[:, :3], rays_train[:, 3:6], is_train=is_train, N_samples=N_samples)
        else:
            xyz_sampled, z_vals, ray_valid = tensorf.module.sample_ray(
                rays_train[:, :3], rays_train[:, 3:6], is_train=is_train, N_samples=N_samples)
    else:
        if ray_type == "ndc":
            xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_ndc(
                rays_train[:, :3], rays_train[:, 3:6], is_train=is_train, N_samples=N_samples)
        elif ray_type == "contract":
            xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_contracted(
                rays_train[:, :3], rays_train[:, 3:6], is_train=is_train, N_samples=N_samples)
        else:
            xyz_sampled, z_vals, ray_valid = tensorf.sample_ray(
                rays_train[:, :3], rays_train[:, 3:6], is_train=is_train, N_samples=N_samples)
    z_vals = torch.tile(z_vals, (xyz_sampled.shape[0], 1))
    return xyz_sampled, z_vals, ray_valid


def raw2outputs(rgb_s,
                sigma_s,
                rgb_d,
                sigma_d,
                dists,
                blending,
                z_vals,
                rays_chunk,
                is_train=False,
                ray_type="ndc"):
    """Transforms model's predictions to semantically meaningful values.
    Args:
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(sigma, dist):
        return 1. - torch.exp(-sigma*dist)

    # # Add noise to model's predictions for density. Can be used to
    # # regularize network during training (prevents floater artifacts).
    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(sigma_d, dists)  # [N_rays, N_samples]
    alpha_s = raw2alpha(sigma_s, dists)  # [N_rays, N_samples]
    alphas = 1. - (1. - alpha_s) * (1. - alpha_d)  # [N_rays, N_samples]

    T_d = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)).to(
        alpha_d.device), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
    T_s = torch.cumprod(torch.cat([torch.ones((alpha_s.shape[0], 1)).to(
        alpha_s.device), 1. - alpha_s + 1e-10], -1), -1)[:, :-1]
    T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)).to(
        alpha_d.device), (1. - alpha_d * blending) * (1. - alpha_s * (1. - blending)) + 1e-10], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device), torch.pow(1. - alpha_d + 1e-10, blending) * torch.pow(1. - alpha_s + 1e-10, 1. - blending)], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)).to(alpha_d.device), (1. - alpha_d) * (1. - alpha_s) + 1e-10], -1), -1)[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_d = weights_d / (torch.sum(weights_d, -1, keepdim=True) + 1e-10)
    # weights_s = weights_s / (torch.sum(weights_s, -1, keepdim=True) + 1e-10)
    weights_full = (alpha_d * blending + alpha_s * (1. - blending)) * T_full
    # weights_full = alphas * T_full

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d +
        (T_full * alpha_s * (1. - blending))[..., None] * rgb_s, -2)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    if is_train and torch.rand((1,)) < 0.5:
        rgb_map_d = rgb_map_d + (1. - acc_map_d[..., None])
        rgb_map_s = rgb_map_s + (1. - acc_map_s[..., None])
        rgb_map_full = rgb_map_full + torch.relu(1. - acc_map_full[..., None])

    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    # depth_map_full = torch.sum(
    #     (T_full * alpha_d * blending) * z_vals + \
    #     (T_full * alpha_s * (1. - blending)) * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)
    if ray_type == "ndc":
        depth_map_d = depth_map_d + (1. - acc_map_d) * (rays_chunk[..., 2] + rays_chunk[..., -1])
        depth_map_s = depth_map_s + (1. - acc_map_s) * (rays_chunk[..., 2] + rays_chunk[..., -1])
        depth_map_full = depth_map_full + torch.relu(1. - acc_map_full) * (rays_chunk[..., 2] + rays_chunk[..., -1])
    elif ray_type == "contract":
        depth_map_d = depth_map_d + (1. - acc_map_d) * 256.0
        depth_map_s = depth_map_s + (1. - acc_map_s) * 256.0
        depth_map_full = depth_map_full + torch.relu(1. - acc_map_full) * 256.0  # TODO: wrong here!

    # TODO: depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1] ?

    rgb_map_d = rgb_map_d.clamp(0, 1)
    rgb_map_s = rgb_map_s.clamp(0, 1)
    rgb_map_full = rgb_map_full.clamp(0, 1)

    # Computed dynamicness
    dynamicness_map = torch.sum(weights_full * blending, -1)
    dynamicness_map = dynamicness_map + torch.relu(1. - acc_map_full) * 0.0
    # dynamicness_map = 1 - T_d[..., -1]

    return rgb_map_full, depth_map_full, acc_map_full, weights_full, \
        rgb_map_s, depth_map_s, acc_map_s, weights_s, \
        rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map


@torch.no_grad()
def evaluation(
    test_dataset,
    poses_mtx,
    focal_ratio_refine,
    tensorf_static,
    tensorf,
    args,
    savePath,
    N_vis,
    N_samples,
    white_bg,
    ray_type,
    compute_metrics,
    writer,
    device,
    make_videos,
):

    N = len(poses_mtx)
    near_fars = []
    metrics = defaultdict(list)

    savePath = Path(savePath)
    savePath.mkdir(parents=True, exist_ok=True)

    img_eval_interval = 1 if N_vis < 0 else N // N_vis + (N % N_vis > 0)
    indices = list(range(0, len(poses_mtx)))[::img_eval_interval]

    W, H = test_dataset.img_wh
    directions = get_ray_directions_blender(
        H, W, [focal_ratio_refine, focal_ratio_refine]).to(poses_mtx.device)  # (H, W, 3)
    if args.multiview_dataset:
        raise NotImplementedError()

    all_depths = defaultdict(list)
    for idx in tqdm(indices):
        rays_o, rays_d = get_rays(directions, poses_mtx[idx])  # both (h*w, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender(H, W, focal_ratio_refine, 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], dim=1)

        W, H = test_dataset.img_wh
        ts = test_dataset.all_ts[idx].view(-1)

        N_rays_all = rays.shape[0]
        chunk = 1024
        chunk_output = defaultdict(list)
        for chunk_idx in range(0, N_rays_all, chunk):
            rays_chunk = rays[chunk_idx:chunk_idx+chunk].to(device)
            ts_chunk = ts[chunk_idx:chunk_idx+chunk].to(device)
            # allposes_refine_train_chunk = allposes_refine_train[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
            xyz_sampled, z_vals, ray_valid = sampleXYZ(
                tensorf, rays_chunk, N_samples=N_samples, ray_type=ray_type, is_train=False, multiGPU=args.multiGPU is not None)
            # static
            _, _, _, _, _, _, rgb_point_static, sigma_static, _, _ = tensorf_static(
                rays_chunk, ts_chunk, None, xyz_sampled, z_vals, ray_valid, is_train=False, white_bg=white_bg, ray_type=ray_type, N_samples=N_samples)
            # dynamic
            _, _, blending, _, _, _, rgb_point_dynamic, sigma_dynamic, z_val_dynamic, dist_dynamic = tensorf(
                rays_chunk, ts_chunk, None, xyz_sampled, z_vals, ray_valid, is_train=False, white_bg=white_bg, ray_type=ray_type, N_samples=N_samples)
            # blending
            rgb_map, depth_map, _, _, \
                rgb_map_s, depth_map_s, _, _, \
                rgb_map_d, depth_map_d, _, _, dynamicness_map = raw2outputs(rgb_point_static, sigma_static, rgb_point_dynamic.to(
                    device), sigma_dynamic.to(device), dist_dynamic.to(device), blending, z_val_dynamic.to(device), rays_chunk, ray_type=ray_type)
            # gather chunks
            chunk_output["rgb_map"].append(rgb_map.cpu().numpy())
            chunk_output["rgb_map_s"].append(rgb_map_s.cpu().numpy())
            chunk_output["rgb_map_d"].append(rgb_map_d.cpu().numpy())
            chunk_output["depth_map"].append(depth_map.cpu().numpy())
            chunk_output["depth_map_s"].append(depth_map_s.cpu().numpy())
            chunk_output["depth_map_d"].append(depth_map_d.cpu().numpy())
            chunk_output["dynamicness_map"].append(dynamicness_map.cpu().numpy())

        chunk_output = {
            k: np.concatenate(v, axis=0).reshape([H, W, -1]) for k, v in chunk_output.items()
        }
        chunk_output = {
            k: v.squeeze(-1) if v.shape[-1] == 1 else v for k, v in chunk_output.items()
        }

        image_keys = [k for k in chunk_output.keys() if k.startswith("rgb_map") or k in ["dynamicness_map"]]
        depth_keys = [k for k in chunk_output.keys() if k.startswith("depth_map")]

        # process data
        for key in image_keys:
            chunk_output[key] = np.clip(chunk_output[key], 0, 1)

        # get bounds
        depth_map_s = chunk_output["depth_map_s"]
        if ray_type == "contract":
            near_fars.append(np.percentile(depth_map_s, [1, 99]))
        else:
            near_fars.append(np.percentile(1/(depth_map_s + 1e-6), [1, 99]))

        # if ray_type == "contract":
        #     for key in depth_keys:
        #         chunk_output[key] = -1/(chunk_output[key] + 1e-6)

        if compute_metrics:
            rgb_map = chunk_output["rgb_map"]
            gt_rgb = test_dataset.all_rgbs[idx].view(H, W, 3).numpy()
            device = tensorf.device if args.multiGPU is None else tensorf.module.device

            loss = float(np.mean((rgb_map - gt_rgb) ** 2))
            metrics["psnr"].append(-10.0 * np.log(loss) / np.log(10.0))
            metrics["ssim"].append(rgb_ssim(rgb_map, gt_rgb, 1))
            metrics["lpips"].append(rgb_lpips(gt_rgb, rgb_map, 'alex', device))
            metrics["vgg"].append(rgb_lpips(gt_rgb, rgb_map, 'vgg', device))

        def save_image(key, is_depth=False):
            folder = savePath / key
            folder.mkdir(parents=True, exist_ok=True)
            data = chunk_output[key]
            if is_depth:
                data = 1 / np.clip(data, 1, 256)
                data = visualize_array(data)

            save_image_np(folder / f"{idx:03d}.png", data)

        for key in image_keys:
            save_image(key)
        for key in depth_keys:
            all_depths[key].append(chunk_output[key])
            save_image(key, True)

    poses_saving = poses_mtx.cpu().numpy()[indices]
    poses_saving = np.concatenate(
        [
            -poses_saving[..., 1:2],
            poses_saving[..., 0:1],
            poses_saving[..., 2:4],
            np.zeros_like(poses_saving[..., 0:1]),
        ],
        axis=-1,
    )
    hwf = np.array([[H, W, focal_ratio_refine]], dtype=np.float32)
    poses_saving[..., 4] = hwf

    poses_saving = poses_saving.reshape([len(poses_saving), 15])
    poses_saving = np.concatenate([poses_saving, np.array(near_fars)], axis=-1)
    np.save(savePath / 'poses_bounds_ours.npy', poses_saving)

    # save globally scaled depths
    all_depth_values = np.array(list(all_depths.values()))
    near, far = np.percentile(all_depth_values, [0.1, 99.9])
    print(f"Visualize globally scaled depths, near={near}, far={far}")

    for key in depth_keys:
        folder = savePath / f"{key}_global"
        folder.mkdir(parents=True, exist_ok=True)
        depth_maps = all_depths[key]
        for idx in range(len(depth_maps)):
            dm = np.clip(depth_maps[idx], near, far)
            dm = (dm - near) / (far - near)
            dm = visualize_normalized_depths(dm)
            save_image_np(folder / f"{idx:03d}.png", dm)

    metrics = {k: float(np.array(v).mean()) for k, v in metrics.items()}
    with open(savePath / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    for key in image_keys + depth_keys + [f"{key}_global" for key in depth_keys]:
        if not make_videos:
            break

        ffmpeg_args = [
            "ffmpeg",
            "-y",
            "-r",
            10,
            "-i",
            savePath / f"{key}/%03d.png",
            "-vf",
            "format=yuv420p",
            "-crf",
            "17",
            "-r",
            30,
            savePath / f"{key}.mp4",
        ]
        ffmpeg_args = [str(v) for v in ffmpeg_args]
        print(" ".join(ffmpeg_args))
        subprocess.run(ffmpeg_args)

    return metrics, near_fars


def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-6) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def world2NDC(pts_world, H, W, f):

    o0 = -1./(W/(2.*f)) * pts_world[..., 0:1] / pts_world[..., 2:]
    o1 = -1./(H/(2.*f)) * pts_world[..., 1:2] / pts_world[..., 2:]
    o2 = 1. + 2. * 1 / pts_world[..., 2:]
    pts = torch.cat([o0, o1, o2], -1)

    return pts


def contract2world(pts_contract):
    # pts_norm = torch.norm(pts_contract.clone(), p=2, dim=-1)
    pts_norm, _ = torch.max(torch.abs(pts_contract.clone()), dim=-1)
    contract_mask = pts_norm > 1.0
    scale = -1/(pts_norm-2)
    pts_world = pts_contract
    pts_world[~contract_mask] = pts_contract[~contract_mask]
    pts_world[contract_mask] = pts_contract[contract_mask] / \
        (pts_norm[contract_mask][:, None]) * scale[contract_mask][:, None]  # TODO: NaN?
    return pts_world


def render_single_3d_point(H, W, f, c2w, pt_NDC):
    """Render 3D position along each ray and project it to the image plane.
    """
    w2c = c2w[:, :3, :3].transpose(1, 2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pt_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[..., 3]
    # Rotate
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)
    # pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                           - pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                          -1)
    # pts_disparity = 1.0 / pts_map_cam[..., 2:]

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, ((pts_map_cam_NDC[:, 2:]+1.0)/2.0)
    # return pts_plane, (((-pts_map_cam_NDC[:, 2:])+1.0)/2.0)
    # return pts_plane, 1./(-pts_map_cam_NDC[:, 2:])


def render_3d_point(H, W, f, c2w, weights, pts, rays, ray_type="ndc"):
    """Render 3D position along each ray and project it to the image plane.
    """
    w2c = c2w[:, :3, :3].transpose(1, 2)
    # w2c = c2w[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])

    # Rendered 3D position in NDC coordinate
    acc_map = torch.sum(weights, -1)[:, None]
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)
    if ray_type == "ndc":
        pts_map_NDC = pts_map_NDC + (1. - acc_map) * (rays[:, :3] + rays[:, 3:])
    elif ray_type == "contract":
        farest_pts = rays[:, :3] + rays[:, 3:] * 256.0
        # convert to contract domain
        # farest_pts_norm = torch.norm(farest_pts.clone(), p=2, dim=-1)
        farest_pts_norm, _ = torch.max(torch.abs(farest_pts.clone()), dim=-1)
        contract_mask = farest_pts_norm > 1.0
        farest_pts[contract_mask] = (2 - 1 / farest_pts_norm[contract_mask])[..., None] * (
            farest_pts[contract_mask] / farest_pts_norm[contract_mask][..., None]
        )
        pts_map_NDC = pts_map_NDC + (1. - acc_map) * farest_pts

    # NDC coordinate to world coordinate
    if ray_type == "ndc":
        pts_map_world = NDC2world(pts_map_NDC, H, W, f)
    elif ray_type == "contract":
        pts_map_world = contract2world(pts_map_NDC)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[..., 3]
    # Rotate
    pts_map_cam = torch.sum(torch.mul(pts_map_world[..., None, :], w2c[:, :3, :3]), -1)
    # pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                           - pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                          -1)
    # pts_disparity = 1.0 / pts_map_cam[..., 2:]

    pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)

    return pts_plane, pts_map_cam_NDC[:, 2:]
    # return pts_plane, (((-pts_map_cam_NDC[:, 2:])+1.0)/2.0)
    # return pts_plane, 1./(-pts_map_cam_NDC[:, 2:])

    # # convert 3D points to NDC
    # pts_map_cam_NDC = world2NDC(pts_map_cam, H, W, f)
    # return pts_plane, pts_map_cam_NDC[:, 2:]


def induce_flow_single(H, W, focal, pose_neighbor, pts_3d_neighbor, pts_2d):

    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor, _ = render_single_3d_point(H, W, focal,
                                                pose_neighbor,
                                                pts_3d_neighbor)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow


def induce_flow(H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d, rays, ray_type="ndc"):

    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor, induced_disp = render_3d_point(H, W, focal,
                                                    pose_neighbor,
                                                    weights,
                                                    pts_3d_neighbor, rays, ray_type)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow, induced_disp
