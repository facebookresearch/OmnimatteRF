# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import io
import os
import sys
from typing import List
import json

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from torch_efficient_distloss import (eff_distloss, eff_distloss_native,
                                      flatten_eff_distloss)
from tqdm.auto import tqdm

from camera import (cam2world, get_novel_view_poses, lie, pose, pose_to_mtx,
                    procrustes_analysis, rotation_distance)
from dataLoader import dataset_dict
from dataLoader.nvidia_pose import center_poses
from dataLoader.ray_utils import (get_ray_directions_blender,
                                  get_ray_directions_lean, get_rays,
                                  get_rays_lean, get_rays_with_batch,
                                  ndc_rays_blender, ndc_rays_blender2)
from flow_viz import flow_to_image
from models.tensoRF import (TensorCP, TensorMMt, TensorVM, TensorVMSplit,
                            TensorVMSplit_TimeEmbedding, TensorVMVt)
from opt import config_parser
from renderer import (NDC2world, OctreeRender_trilinear_fast, contract2world,
                      evaluation, induce_flow,
                      induce_flow_single, raw2outputs, render_3d_point,
                      render_single_3d_point, sampleXYZ)
from utils import N_to_reso, TVLoss, cal_n_samples, convert_sdf_samples_to_ply, save_image_np
from pathlib import Path
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


def ids2pixel(W, H, ids):
    """
    Regress pixel coordinates from
    """
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids


@torch.no_grad()
def export_mesh(args):

    ckpt = None
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def draw_poses(pose_aligned, gt_poses):
    # matplotlib poses visualization
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if gt_poses is not None:
        vertices, faces, wireframe = get_camera_mesh(gt_poses, 0.005)
        center_gt = vertices[:, -1]
        ax.scatter(center_gt[:, 0], center_gt[:, 1], center_gt[:, 2], marker='o', color='C0')
        wireframe_merged = merge_wireframes(wireframe)
        for c in range(center_gt.shape[0]):
            ax.plot(wireframe_merged[0][c*10:(c+1)*10], wireframe_merged[1]
                    [c*10:(c+1)*10], wireframe_merged[2][c*10:(c+1)*10], color='C0')

    vertices, faces, wireframe = get_camera_mesh(pose_aligned.cpu(), 0.005)
    center = vertices[:, -1]
    ax.scatter(center[:, 0], center[:, 1], center[:, 2], marker='o', color='C1')
    wireframe_merged = merge_wireframes(wireframe)
    for c in range(center.shape[0]):
        ax.plot(wireframe_merged[0][c*10:(c+1)*10], wireframe_merged[1]
                [c*10:(c+1)*10], wireframe_merged[2][c*10:(c+1)*10], color='C1')

    if gt_poses is not None:
        for i in range(center.shape[0]):
            ax.plot([center_gt[i, 0], center[i, 0]], [center_gt[i, 1], center[i, 1]],
                    [center_gt[i, 2], center[i, 2]], color='red')

    set_axes_equal(ax)
    plt.tight_layout()

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img.astype(np.float32) / 255
    plt.close(fig)
    return img


# from DynDyn


def generate_path(c2w, focal, sc):
    # hwf = c2w[:, 4:5]
    num_novelviews = 60
    max_disp = 48.0*2
    # H, W, focal = hwf[:, 0]
    # downsample = 2.0
    # focal = (854 / 2 * np.sqrt(3)) / float(downsample)

    max_trans = max_disp / focal[0] * sc
    output_poses = []
    output_focals = []

    # Rendering teaser. Add translation.
    for i in range(num_novelviews):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * 1.0
        y_trans = max_trans * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews)) - 1.) * 0.33
        z_trans = 0.

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        output_poses.append(render_pose[:3, :])
        # output_focals.append(focal)
    return output_poses


@torch.no_grad()
def render_test(args):
    ray_type = args.ray_type
    logfolder = f'{args.basedir}/{args.expname}'

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train,
                           is_stack=True, use_disp=args.use_disp, use_foreground_mask=args.use_foreground_mask)
    white_bg = test_dataset.white_bg

    if not os.path.exists(args.ckpt):
        raise RuntimeError('the ckpt path does not exists!!')

    # dynamic
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    poses_mtx = kwargs.pop("se3_poses").to(device)
    focal_refine = kwargs.pop("focal_ratio_refine").to(device)
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    if args.multiGPU is not None:
        tensorf = torch.nn.DataParallel(tensorf, device_ids=args.multiGPU)
    # static
    ckpt_static = torch.load(args.ckpt[:-3]+'_static.th', map_location=device)
    kwargs_static = ckpt_static['kwargs']
    poses_mtx = kwargs_static.pop("se3_poses").to(device)
    focal_refine = kwargs_static.pop("focal_ratio_refine").to(device)
    kwargs_static.update({'device': device})
    tensorf_static = TensorVMSplit(**kwargs_static)
    tensorf_static.load(ckpt_static)
    if args.multiGPU is not None:
        tensorf_static = torch.nn.DataParallel(tensorf_static, device_ids=args.multiGPU)

    iteration = Path(args.ckpt).stem
    focal_refine_value = focal_refine.cpu()
    print(f"focal_refine: {focal_refine}")

    savePath = Path(f"{logfolder}/render_test/{iteration}")
    savePath.mkdir(parents=True, exist_ok=True)
    poses_img = draw_poses(poses_mtx, None)
    save_image_np(savePath / "camera_poses.png", poses_img)

    test_metrics, near_fars = evaluation(
        test_dataset,
        poses_mtx,
        focal_refine_value,
        tensorf_static,
        tensorf,
        args,
        f"{logfolder}/render_test/{iteration}",
        N_vis=-1,
        N_samples=-1,
        white_bg=white_bg,
        ray_type=ray_type,
        compute_metrics=True,
        writer=None,
        device=device,
        make_videos=True,
    )

    if args.render_path:
        SE3_poses = poses_mtx

        # find a better focal distance
        # center pose
        render_idx = SE3_poses.shape[0]//2 - 1
        sc = (near_fars[render_idx][0] * 0.75)
        c2w = SE3_poses.cpu().detach().numpy()[render_idx]

        # Get average pose
        up_m = normalize(SE3_poses.cpu().detach().numpy()[:, :3, 1].sum(0))

        c2ws = generate_path(SE3_poses.cpu().detach().numpy()[render_idx], focal=[
                             focal_refine.item(), focal_refine.item()], sc=sc)
        c2ws = np.stack(c2ws, 0)[:, :3]

        test_metrics, _ = evaluation(
            test_dataset,
            c2ws,
            focal_refine_value,
            tensorf_static,
            tensorf,
            args,
            f"{logfolder}/render_path/{iteration}",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=ray_type,
            compute_metrics=False,
            writer=None,
            device=device,
            make_videos=True,
        )


@torch.no_grad()
def prealign_cameras(pose_in, pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3, device=pose_in.device)
    center_pred = cam2world(center, pose_in)[:, 0]  # [N,3]
    center_GT = cam2world(center, pose_GT)[:, 0]  # [N,3]
    try:
        sim3 = procrustes_analysis(center_GT, center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=pose_in.device))
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose_in[..., :3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[..., None])[..., 0]
    pose_aligned = pose(R=R_aligned, t=t_aligned)
    return pose_aligned, sim3


@torch.no_grad()
def evaluate_camera_alignment(pose_aligned, pose_GT):
    # measure errors in rotation and translation
    # pose_aligned: [N, 3, 4]
    # pose_GT:      [N, 3, 4]
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned-t_GT)[..., 0].norm(dim=-1)
    return R_error, t_error


def get_camera_mesh(pose, depth=1):
    vertices = torch.tensor([[-0.5, -0.5, 1],
                            [0.5, -0.5, 1],
                            [0.5, 0.5, 1],
                            [-0.5, 0.5, 1],
                            [0, 0, 0]])*depth
    faces = torch.tensor([[0, 1, 2],
                          [0, 2, 3],
                          [0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [3, 0, 4]])
    # vertices = cam2world(vertices[None],pose)
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged


def compute_depth_loss(dyn_depth, gt_depth):

    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / (s_d + 1e-10)

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / (s_gt + 1e-10)

    # return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)
    return torch.sum((dyn_depth_norm - gt_depth_norm) ** 2)


def get_stats(X, norm=2):
    """
    :param X (N, H, W, C)
    :returns mean (1, 1, 1, C), scale (1)
    """
    mean = X.mean(dim=(0, 1, 2), keepdim=True)  # (1, 1, 1, C)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=-1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=-1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


def reconstruction(args):
    ray_type = args.ray_type

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, use_disp=args.use_disp,
                            use_foreground_mask=args.use_foreground_mask, with_GT_poses=args.with_GT_poses, ray_type=ray_type)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, use_disp=args.use_disp,
                           use_foreground_mask=args.use_foreground_mask, with_GT_poses=args.with_GT_poses, ray_type=ray_type)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    W, H = train_dataset.img_wh

    # set N_voxel_t to match dataset length
    if args.N_voxel_t < 0:
        args.N_voxel_t = len(test_dataset.all_rgbs)

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log fileinit log file
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # save args
    with open(logfolder + "/args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # static TensoRF
    tensorf_static = TensorVMSplit(aabb, reso_cur, args.N_voxel_t, device,
                                   density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                                   shadingMode=args.shadingModeStatic, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                                   pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)
    if args.multiGPU is not None:
        tensorf_static = torch.nn.DataParallel(tensorf_static, device_ids=args.multiGPU)

    # dynamic tensorf
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, args.N_voxel_t, device,
                                        density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                                        shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                                        pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    if args.multiGPU is not None:
        tensorf = torch.nn.DataParallel(tensorf, device_ids=args.multiGPU)

    if args.use_time_embedding:
        time_embeddings = torch.nn.Embedding(args.N_voxel_t, args.time_embedding_size).to(device)

    grad_vars = tensorf_static.get_optparam_groups(
        args.lr_init, args.lr_basis) if args.multiGPU is None else tensorf_static.module.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars.extend(tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
                     if args.multiGPU is None else tensorf.module.get_optparam_groups(args.lr_init, args.lr_basis))
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    if args.use_time_embedding:
        grad_vars += [{'params': time_embeddings.parameters(), 'lr': args.lr_init}]

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrgbs = train_dataset.all_rgbs
    allts = train_dataset.all_ts
    if args.with_GT_poses:
        allposes = train_dataset.all_poses  # (12, 3, 4)
    else:
        allposes = None
    allflows_f = train_dataset.all_flows_f.to(device)
    allflowmasks_f = train_dataset.all_flow_masks_f.to(device)
    allflows_b = train_dataset.all_flows_b.to(device)
    allflowmasks_b = train_dataset.all_flow_masks_b.to(device)

    if args.use_disp:
        alldisps = train_dataset.all_disps
    # if args.use_foreground_mask:
    allforegroundmasks = train_dataset.all_foreground_masks

    init_poses = torch.zeros(args.N_voxel_t, 9)
    if args.with_GT_poses:
        init_poses[..., 0:3] = allposes[..., :, 0]
        init_poses[..., 3:6] = allposes[..., :, 1]
        init_poses[..., 6:9] = allposes[..., :, 3]
    else:
        init_poses[..., 0] = 1
        init_poses[..., 4] = 1
    poses_refine = torch.nn.Embedding(args.N_voxel_t, 9).to(device)
    poses_refine.weight.data.copy_(init_poses.to(device))

    # optimizing focal length
    print(f"initialize FOV to {args.fov_init} degrees")
    fov_refine_embedding = torch.nn.Embedding(1, 1).to(device)
    fov_refine_embedding.weight.data.copy_(torch.ones(1, 1).to(device) * args.fov_init / 180 * np.pi)
    if args.with_GT_poses:
        focal_refine = torch.tensor(train_dataset.focal[0]).to(device)

    ii, jj = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    grid = torch.from_numpy(np.stack([ii, jj], -1)).to(device)
    grid = torch.tile(torch.unsqueeze(grid, 0), (args.N_voxel_t, 1, 1, 1))
    allgrids = grid.view(-1, 2)

    # setup optimizer
    if args.optimize_poses:
        lr_pose = 3e-3
        lr_pose_end = 1e-5  # 5:X, 10:X
        optimizer_pose = torch.optim.Adam(poses_refine.parameters(), lr=lr_pose)
        gamma = (lr_pose_end/lr_pose)**(1./(args.n_iters - args.upsamp_list[-1]))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_pose, gamma=gamma)

    if args.optimize_focal_length:
        lr_pose = 3e-3
        lr_pose_end = 1e-5  # 5:X, 10:X
        optimizer_focal = torch.optim.Adam(fov_refine_embedding.parameters(), lr=0.0)
        gamma = (lr_pose_end/lr_pose)**(1./(args.n_iters - args.upsamp_list[-1]))
        scheduler_focal = torch.optim.lr_scheduler.ExponentialLR(optimizer_focal, gamma=gamma)

    optimizing_frames = int(np.sqrt(args.N_voxel_t))
    init_steps = int(args.upsamp_list[0] / optimizing_frames)

    trainingSampler = SimpleSampler(allts.shape[0], args.batch_size)
    trainingSampler_2 = SimpleSampler(allts.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    distortion_weight_static, distortion_weight_dynamic = args.distortion_weight_static, args.distortion_weight_dynamic
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    decay_iteration = 100

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)

    for iteration in pbar:

        def write_metrics(name, metrics):
            print(f"======> {args.expname} {name} metrics ({iteration+1}) <======")
            for k, v in metrics.items():
                print(f"{k}: {v}")
                summary_writer.add_scalar(f"{name}/{k}", v, global_step=iteration)

        # Lambda decay.
        Temp_static = 1. / (10 ** (iteration / (100000)))
        Temp = 1. / (10 ** (iteration // (decay_iteration * 1000)))
        Temp_disp_TV = 1. / (10 ** (iteration // (50000)))

        if args.optimize_focal_length:
            focal_refine = np.maximum(H, W) / 2.0 / torch.tan(fov_refine_embedding.weight[0, 0])

        ray_idx = trainingSampler.nextids()

        # rays_train, rgb_train, ts_train = allrays[ray_idx], allrgbs[ray_idx].to(device), allts[ray_idx]

        rgb_train, ts_train, grid_train = allrgbs[ray_idx].to(device), allts[ray_idx], allgrids[ray_idx]
        flow_f_train, flow_mask_f_train, flow_b_train, flow_mask_b_train = allflows_f[ray_idx], allflowmasks_f[
            ray_idx][..., None], allflows_b[ray_idx], allflowmasks_b[ray_idx][..., None]
        # flow_for_grouping_train = flow_for_grouping[ray_idx]

        if args.use_time_embedding:
            if args.multiview_dataset:
                all_time_embeddings = torch.tile(torch.unsqueeze(torch.unsqueeze(
                    time_embeddings.weight, 1), 1), (1, args.N_voxel_t, W*H, 1))  # (time, view, H*W, 6)
            else:
                all_time_embeddings = torch.tile(torch.unsqueeze(
                    torch.unsqueeze(time_embeddings.weight, 1), 1), (1, 1, W*H, 1))
            all_time_embeddings = all_time_embeddings.view(-1, args.time_embedding_size)
            alltimeembedding = all_time_embeddings[ray_idx]
        else:
            alltimeembedding = None

        if args.use_disp:
            alldisps_train = alldisps[ray_idx].to(device)

        allforegroundmasks_train = allforegroundmasks[ray_idx].to(device)

        # allposes_refine = lie.se3_to_SE3(se3_refine.weight)
        poses_refine2 = poses_refine.weight.clone()
        poses_refine2[..., 6:9] = poses_refine2[..., 6:9]
        poses_mtx = pose_to_mtx(poses_refine2)

        i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))

        directions = get_ray_directions_lean(i, j, [focal_refine, focal_refine], [W / 2, H / 2])
        poses_mtx_batched = poses_mtx[view_ids]
        rays_o, rays_d = get_rays_lean(directions, poses_mtx_batched)  # both (b, 3)
        if ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender2(H, W, [focal_refine, focal_refine], 1.0, rays_o, rays_d)
        rays_train = torch.cat([rays_o, rays_d], -1).view(-1, 6)

        # index the pose for forward and backward
        allposes_refine_f = torch.cat((poses_mtx[1:], poses_mtx[-1:]), 0)
        allposes_refine_b = torch.cat((poses_mtx[0:1], poses_mtx[:-1]), 0)
        allposes_refine_f_train = torch.tile(torch.unsqueeze(torch.unsqueeze(
            allposes_refine_f, 1), 1), (1, H, W, 1, 1)).view(-1, 3, 4)[ray_idx]
        allposes_refine_b_train = torch.tile(torch.unsqueeze(torch.unsqueeze(
            allposes_refine_b, 1), 1), (1, H, W, 1, 1)).view(-1, 3, 4)[ray_idx]

        t_ref = ray_idx // (H*W)
        u_ref = (ray_idx % (H*W)) // W  # height
        v_ref = (ray_idx % (H*W)) % W  # width
        t_interval = 2 / (args.N_voxel_t - 1)

        total_loss = 0.0

        # static part for pose estimation
        xyz_sampled, z_vals, ray_valid = sampleXYZ(
            tensorf, rays_train, N_samples=nSamples, ray_type=ray_type, is_train=True, multiGPU=args.multiGPU is not None)
        # static tensorf
        _, _, _, pts_ref_s, _, _, rgb_points_static, sigmas_static, _, _ = tensorf_static(
            rays_train, ts_train, None, xyz_sampled, z_vals, ray_valid, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
        # dynamic tensorf
        _, _, blending, pts_ref, _, _, rgb_points_dynamic, sigmas_dynamic, z_vals_dynamic, dists_dynamic = tensorf(
            rays_train, ts_train, None, xyz_sampled, z_vals, ray_valid, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
        _, _, _, _, \
            rgb_map_s, depth_map_s, _, weights_s, \
            _, _, _, _, _ = raw2outputs(rgb_points_static, sigmas_static, rgb_points_dynamic, sigmas_dynamic,
                                        dists_dynamic, blending, z_vals_dynamic, rays_train, is_train=True, ray_type=ray_type)

        # static losses
        # RGB loss
        img_s_loss = torch.sum((rgb_map_s - rgb_train) ** 2 * (1.0 - allforegroundmasks_train[..., 0:1])) / (
            torch.sum((1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8) / rgb_map_s.shape[-1]
        total_loss += (1.0*img_s_loss)
        summary_writer.add_scalar('train/img_s_loss', img_s_loss.detach().item(), global_step=iteration)

        # static distortion loss from DVGO
        if distortion_weight_static > 0:
            ray_id = torch.tile(torch.range(0, args.batch_size-1, dtype=torch.int64)
                                [:, None], (1, weights_s.shape[1])).to(device)
            loss_distortion_static = flatten_eff_distloss(torch.flatten(
                weights_s), torch.flatten(z_vals), 1/(weights_s.shape[1]), torch.flatten(ray_id))
            total_loss += (loss_distortion_static*distortion_weight_static*(iteration/args.n_iters))
            summary_writer.add_scalar('train/loss_distortion_static',
                                      (loss_distortion_static).detach().item(), global_step=iteration)

        if L1_reg_weight > 0:
            loss_reg_L1_density_s = tensorf_static.density_L1() if args.multiGPU is None else tensorf_static.module.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1_density_s
            summary_writer.add_scalar('train/loss_reg_L1_density_s',
                                      loss_reg_L1_density_s.detach().item(), global_step=iteration)

        if TV_weight_density > 0:
            if args.multiGPU is None:
                loss_tv_static = tensorf_static.TV_loss_density(tvreg) * TV_weight_density
                # loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density * (iteration/(args.n_iters-1.0))
            else:
                loss_tv_static = tensorf_static.module.TV_loss_density(tvreg) * TV_weight_density
                # loss_tv = tensorf.module.TV_loss_density(tvreg) * TV_weight_density * (iteration/(args.n_iters-1.0))
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar('train/reg_tv_density_static',
                                      loss_tv_static.detach().item(), global_step=iteration)
        if TV_weight_app > 0:
            if args.multiGPU is None:
                loss_tv_static = tensorf_static.TV_loss_app(tvreg)*TV_weight_app
                # loss_tv = loss_tv + tensorf.TV_loss_app(tvreg)*TV_weight_app * (iteration/(args.n_iters-1.0))
            else:
                loss_tv_static = tensorf_static.module.TV_loss_app(tvreg)*TV_weight_app
                # loss_tv = loss_tv + tensorf.module.TV_loss_app(tvreg)*TV_weight_app * (iteration/(args.n_iters-1.0))
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar('train/reg_tv_app_static', loss_tv_static.detach().item(), global_step=iteration)

        summary_writer.add_scalar('train/focal_ratio_refine', focal_refine.detach().item(), global_step=iteration)

        if args.optimize_poses:
            # static motion loss
            induced_flow_f_s, induced_disp_f_s = induce_flow(
                H, W, focal_refine, allposes_refine_f_train, weights_s, pts_ref_s, grid_train, rays_train, ray_type=ray_type)
            flow_f_s_loss = torch.sum(torch.abs(induced_flow_f_s - flow_f_train) * flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])) / (
                torch.sum(flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8) / flow_f_train.shape[-1]
            total_loss += (0.02*flow_f_s_loss*Temp_static)
            induced_flow_b_s, induced_disp_b_s = induce_flow(
                H, W, focal_refine, allposes_refine_b_train, weights_s, pts_ref_s, grid_train, rays_train, ray_type=ray_type)
            flow_b_s_loss = torch.sum(torch.abs(induced_flow_b_s - flow_b_train) * flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])) / (
                torch.sum(flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8) / flow_b_train.shape[-1]
            total_loss += (0.02*flow_b_s_loss*Temp_static)
            summary_writer.add_scalar('train/flow_f_s_loss', flow_f_s_loss.detach().item(), global_step=iteration)
            summary_writer.add_scalar('train/flow_b_s_loss', flow_b_s_loss.detach().item(), global_step=iteration)

            # static disparity loss
            # forward
            uv_f = torch.stack((v_ref+0.5, u_ref+0.5), -1).to(flow_f_train.device) + flow_f_train
            directions_f = torch.stack([(uv_f[..., 0] - W / 2) / (focal_refine), -(uv_f[..., 1] -
                                       H / 2) / (focal_refine), -torch.ones_like(uv_f[..., 0])], -1)  # (H, W, 3)

            rays_f_o, rays_f_d = get_rays_lean(directions_f, allposes_refine_f_train)  # both (b, 3)
            if ray_type == "ndc":
                rays_f_o, rays_f_d = ndc_rays_blender2(H, W, [focal_refine, focal_refine], 1.0, rays_f_o, rays_f_d)

            rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)
            xyz_sampled_f, z_vals_f, ray_valid_f = sampleXYZ(
                tensorf_static, rays_f_train, N_samples=nSamples, ray_type=ray_type, is_train=True, multiGPU=args.multiGPU is not None)
            _, _, _, pts_ref_s_ff, weights_s_ff, _, _, _, _, _ = tensorf_static(
                rays_f_train, ts_train, None, xyz_sampled_f, z_vals_f, ray_valid_f, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
            _, induced_disp_s_ff = induce_flow(H, W, focal_refine, allposes_refine_f_train,
                                               weights_s_ff, pts_ref_s_ff, grid_train, rays_f_train, ray_type=ray_type)
            disp_f_s_loss = torch.sum(torch.abs(induced_disp_f_s - induced_disp_s_ff) * flow_mask_f_train * (
                1.0 - allforegroundmasks_train[..., 0:1])) / (torch.sum(flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8)
            total_loss += (0.04*disp_f_s_loss*Temp_static)
            summary_writer.add_scalar('train/disp_f_s_loss', disp_f_s_loss.detach().item(), global_step=iteration)
            # backward
            uv_b = torch.stack((v_ref+0.5, u_ref+0.5), -1).to(flow_b_train.device) + flow_b_train
            directions_b = torch.stack([(uv_b[..., 0] - W / 2) / (focal_refine), -(uv_b[..., 1] -
                                       H / 2) / (focal_refine), -torch.ones_like(uv_b[..., 0])], -1)  # (H, W, 3)
            rays_b_o, rays_b_d = get_rays_lean(directions_b, allposes_refine_b_train)  # both (b, 3)
            if ray_type == "ndc":
                rays_b_o, rays_b_d = ndc_rays_blender2(H, W, [focal_refine, focal_refine], 1.0, rays_b_o, rays_b_d)
            rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)
            xyz_sampled_b, z_vals_b, ray_valid_b = sampleXYZ(
                tensorf_static, rays_b_train, N_samples=nSamples, ray_type=ray_type, is_train=True, multiGPU=args.multiGPU is not None)
            _, _, _, pts_ref_s_bb, weights_s_bb, _, _, _, _, _ = tensorf_static(
                rays_b_train, ts_train, None, xyz_sampled_b, z_vals_b, ray_valid_b, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)

            _, induced_disp_s_bb = induce_flow(H, W, focal_refine, allposes_refine_b_train,
                                               weights_s_bb, pts_ref_s_bb, grid_train, rays_b_train, ray_type=ray_type)
            disp_b_s_loss = torch.sum(torch.abs(induced_disp_b_s - induced_disp_s_bb) * flow_mask_b_train * (
                1.0 - allforegroundmasks_train[..., 0:1])) / (torch.sum(flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8)
            total_loss += (0.04*disp_b_s_loss*Temp_static)
            summary_writer.add_scalar('train/disp_b_s_loss', disp_b_s_loss.detach().item(), global_step=iteration)

            # # Monocular depth loss with mask for static TensoRF
            total_mono_depth_loss = 0.0
            counter = 0.0
            for cam_idx in range(args.N_voxel_t):
                valid = torch.bitwise_and(t_ref == cam_idx, allforegroundmasks_train[..., 0].cpu() < 0.5)
                # if valid.any():
                if torch.sum(valid) > 1.0:
                    if ray_type == 'ndc':
                        total_mono_depth_loss += compute_depth_loss(depth_map_s[valid], -alldisps_train[valid])
                    elif ray_type == 'contract':
                        total_mono_depth_loss += compute_depth_loss(1./(depth_map_s[valid]+1e-6), alldisps_train[valid])
                    counter += torch.sum(valid)
            total_mono_depth_loss = total_mono_depth_loss / counter
            total_loss += (total_mono_depth_loss*args.monodepth_weight_static*Temp_static)
            summary_writer.add_scalar('train/total_mono_depth_loss',
                                      total_mono_depth_loss.detach().item(), global_step=iteration)

            # sample for patch TV loss
            i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))
            i_neighbor = torch.clamp(i + 1, max=W - 1)
            j_neighbor = torch.clamp(j + 1, max=H - 1)

            directions_i_neighbor = get_ray_directions_lean(i_neighbor, j, [focal_refine, focal_refine], [W / 2, H / 2])
            rays_o_i_neighbor, rays_d_i_neighbor = get_rays_lean(
                directions_i_neighbor, poses_mtx_batched)  # both (b, 3)
            if ray_type == "ndc":
                rays_o_i_neighbor, rays_d_i_neighbor = ndc_rays_blender2(
                    H, W, [focal_refine, focal_refine], 1.0, rays_o_i_neighbor, rays_d_i_neighbor)
            rays_train_i_neighbor = torch.cat([rays_o_i_neighbor, rays_d_i_neighbor], -1).view(-1, 6)
            directions_j_neighbor = get_ray_directions_lean(i, j_neighbor, [focal_refine, focal_refine], [W / 2, H / 2])
            rays_o_j_neighbor, rays_d_j_neighbor = get_rays_lean(
                directions_j_neighbor, poses_mtx_batched)  # both (b, 3)
            if ray_type == "ndc":
                rays_o_j_neighbor, rays_d_j_neighbor = ndc_rays_blender2(
                    H, W, [focal_refine, focal_refine], 1.0, rays_o_j_neighbor, rays_d_j_neighbor)
            rays_train_j_neighbor = torch.cat([rays_o_j_neighbor, rays_d_j_neighbor], -1).view(-1, 6)
            xyz_sampled_i_neighbor, z_vals_i_neighbor, ray_valid_i_neighbor = sampleXYZ(
                tensorf, rays_train_i_neighbor, N_samples=nSamples, ray_type=ray_type, is_train=True, multiGPU=args.multiGPU is not None)
            _, _, _, pts_ref_s_i_neighbor, _, _, rgb_points_static_i_neighbor, sigmas_static_i_neighbor, _, _ = tensorf_static(
                rays_train_i_neighbor, ts_train, None, xyz_sampled_i_neighbor, z_vals_i_neighbor, ray_valid_i_neighbor, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
            _, _, blending_i_neighbor, pts_ref_i_neighbor, _, _, rgb_points_dynamic_i_neighbor, sigmas_dynamic_i_neighbor, z_vals_dynamic_i_neighbor, dists_dynamic_i_neighbor = tensorf(
                rays_train_i_neighbor, ts_train, None, xyz_sampled_i_neighbor, z_vals_i_neighbor, ray_valid_i_neighbor, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
            _, _, _, _, \
                _, depth_map_s_i_neighbor, _, _, \
                _, _, _, _, _ = raw2outputs(rgb_points_static_i_neighbor, sigmas_static_i_neighbor, rgb_points_dynamic_i_neighbor, sigmas_dynamic_i_neighbor,
                                            dists_dynamic_i_neighbor, blending_i_neighbor, z_vals_dynamic_i_neighbor, rays_train_i_neighbor, is_train=True, ray_type=ray_type)
            xyz_sampled_j_neighbor, z_vals_j_neighbor, ray_valid_j_neighbor = sampleXYZ(
                tensorf, rays_train_j_neighbor, N_samples=nSamples, ray_type=ray_type, is_train=True, multiGPU=args.multiGPU is not None)
            _, _, _, pts_ref_s_j_neighbor, _, _, rgb_points_static_j_neighbor, sigmas_static_j_neighbor, _, _ = tensorf_static(
                rays_train_j_neighbor, ts_train, None, xyz_sampled_j_neighbor, z_vals_j_neighbor, ray_valid_j_neighbor, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
            _, _, blending_j_neighbor, pts_ref_j_neighbor, _, _, rgb_points_dynamic_j_neighbor, sigmas_dynamic_j_neighbor, z_vals_dynamic_j_neighbor, dists_dynamic_j_neighbor = tensorf(
                rays_train_j_neighbor, ts_train, None, xyz_sampled_j_neighbor, z_vals_j_neighbor, ray_valid_j_neighbor, is_train=True, white_bg=white_bg, ray_type=ray_type, N_samples=nSamples)
            _, _, _, _, \
                _, depth_map_s_j_neighbor, _, _, \
                _, _, _, _, _ = raw2outputs(rgb_points_static_j_neighbor, sigmas_static_j_neighbor, rgb_points_dynamic_j_neighbor, sigmas_dynamic_j_neighbor,
                                            dists_dynamic_j_neighbor, blending_j_neighbor, z_vals_dynamic_j_neighbor, rays_train_j_neighbor, is_train=True, ray_type=ray_type)
            disp_smooth_loss = torch.mean(((1.0/torch.clamp(depth_map_s, min=1e-6)) - (1.0/torch.clamp(depth_map_s_i_neighbor, min=1e-6)))**2) + \
                torch.mean(((1.0/torch.clamp(depth_map_s, min=1e-6)) -
                           (1.0/torch.clamp(depth_map_s_j_neighbor, min=1e-6)))**2)
            total_loss += (disp_smooth_loss*50.0*Temp_disp_TV)
            summary_writer.add_scalar('train/disp_smooth_loss', disp_smooth_loss.detach().item(), global_step=iteration)

        if args.optimize_poses:
            optimizer_pose.zero_grad()
        if args.optimize_focal_length:
            optimizer_focal.zero_grad()
        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()
        if args.optimize_poses:
            optimizer_pose.step()
            scheduler.step()
        if args.optimize_focal_length:
            optimizer_focal.step()
            scheduler_focal.step()

        pose_aligned = poses_mtx.clone().detach()

        summary_writer.add_scalar('train/density_app_plane_lr', optimizer.param_groups[0]['lr'], global_step=iteration)
        summary_writer.add_scalar('train/basis_mat_lr', optimizer.param_groups[4]['lr'], global_step=iteration)
        if args.optimize_poses:
            summary_writer.add_scalar('train/lr_pose', optimizer_pose.param_groups[0]['lr'], global_step=iteration)
        if args.optimize_focal_length:
            summary_writer.add_scalar('train/lr_focal', optimizer_focal.param_groups[0]['lr'], global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            if args.with_GT_poses:
                pbar.set_description(f'Iteration {iteration:05d}:'
                                     + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                                     + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}')
            else:
                pbar.set_description(f'Iteration {iteration:05d}:')
            PSNRs = []

            img = draw_poses(pose_aligned, allposes)
            summary_writer.add_image('camera_poses', np.transpose(img, (2, 0, 1)), iteration)

        def check_interval(sancheck, every):
            return (sancheck and iteration == 0) or \
                every > 0 and (iteration + 1) % every == 0

        focal_refine_value = float(focal_refine)
        if args.N_vis > 0 and check_interval(args.vis_train_sancheck, args.vis_train_every):
            print("===== Visualize train =====")
            vis_root = os.path.join(logfolder, f"vis/iter_{iteration+1}")
            Path(vis_root).mkdir(parents=True, exist_ok=True)

            val_metrics, _ = evaluation(
                test_dataset,
                poses_mtx,
                focal_refine_value,
                tensorf_static,
                tensorf,
                args,
                vis_root,
                args.N_vis,
                nSamples,
                white_bg=white_bg,
                ray_type=ray_type,
                compute_metrics=True,
                writer=summary_writer,
                device=device,
                make_videos=False,
            )
            write_metrics("validation", val_metrics)

        if check_interval(args.vis_full_sancheck, args.vis_full_every):
            print("===== Visualize full =====")

            test_metrics, near_fars = evaluation(
                test_dataset,
                poses_mtx,
                focal_refine_value,
                tensorf_static,
                tensorf,
                args,
                f"{logfolder}/render_test/{iteration+1}",
                N_vis=-1,
                N_samples=-1,
                white_bg=white_bg,
                ray_type=ray_type,
                compute_metrics=True,
                writer=None,
                device=device,
                make_videos=True,
            )
            write_metrics("test_all", test_metrics)

            if args.render_path:
                print("===== Visualize path =====")
                render_idx = poses_mtx.shape[0]//2 - 1
                sc = near_fars[render_idx][0] * 0.75

                c2ws = generate_path(
                    poses_mtx.cpu().detach().numpy()[render_idx],
                    focal=[focal_refine_value, focal_refine_value],
                    sc=sc,
                )
                c2ws = torch.from_numpy(np.stack(c2ws, 0)[:, :3].astype(np.float32))

                test_metrics, _ = evaluation(
                    test_dataset,
                    c2ws,
                    focal_refine_value,
                    tensorf_static,
                    tensorf,
                    args,
                    f"{logfolder}/render_path/{iteration+1}",
                    N_vis=-1,
                    N_samples=-1,
                    white_bg=white_bg,
                    ray_type=ray_type,
                    compute_metrics=False,
                    writer=None,
                    device=device,
                    make_videos=True,
                )

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb if args.multiGPU is None else tensorf.module.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            if args.multiGPU is None:
                tensorf.upsample_volume_grid(reso_cur)
                tensorf_static.upsample_volume_grid(reso_cur)
            else:
                tensorf.module.upsample_volume_grid(reso_cur)
                tensorf_static.module.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
                if args.optimize_poses:
                    optimizer_pose.param_groups[0]['lr'] = lr_pose
                if iteration >= args.upsamp_list[3] and args.optimize_focal_length:
                    optimizer_focal.param_groups[0]['lr'] = lr_pose
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf_static.get_optparam_groups(
                args.lr_init*lr_scale, args.lr_basis*lr_scale) if args.multiGPU is None else tensorf_static.module.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            grad_vars.extend(tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
                             if args.multiGPU is None else tensorf.module.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale))
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if (iteration == 0 and args.save_sancheck) or (iteration + 1) % args.save_every == 0:
            print("===== Saving checkpoint =====")
            ckpt_root = Path(logfolder) / "checkpoints"
            ckpt_root.mkdir(parents=True, exist_ok=True)

            tensorf.module.save(
                poses_mtx.detach().cpu(),
                focal_refine.detach().cpu(),
                ckpt_root / f"{iteration+1}.th"
            )
            tensorf_static.module.save(
                poses_mtx.detach().cpu(),
                focal_refine.detach().cpu(),
                ckpt_root / f"{iteration+1}_static.th"
            )


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only:
        render_test(args)
    else:
        reconstruction(args)
