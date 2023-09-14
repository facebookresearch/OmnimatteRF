# Copyright (c) Meta Platforms, Inc. and affiliates.

import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--tblogdir", type=str,
                        help='where to store tensorboard logs, defaulting to basedir')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP', 'TensorVMVt', 'TensorMMt', 'TensorVMSplit_TimeEmbedding'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'nvidia', 'nvidia_pose', 'nvidia_multiview', 'nvidia_multiview_pose', 'nsvf', 'dtu','tankstemple', 'own_data'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--distortion_weight_static", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--distortion_weight_dynamic", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--monodepth_weight_static", type=float, default=0.04,
                        help='loss weight')
    parser.add_argument("--monodepth_weight_dynamic", type=float, default=0.04,
                        help='loss weight')
    parser.add_argument("--smooth_scene_flow_weight", type=float, default=0.1,
                        help='loss weight')
    parser.add_argument("--small_scene_flow_weight", type=float, default=0.1,
                        help='loss weight')

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')

    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--shadingModeStatic", type=str, default="MLP_Fea_TimeEmbedding",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')



    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)
    parser.add_argument("--no_tensorboard", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    # parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument(
        "--ray_type",
        type=str,
        default="ndc",
        choices=["ndc", "contract"],
    )
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)


    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')



    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument('--N_voxel_t',
                        type=int,
                        default=12)
    parser.add_argument("--upsamp_list", type=int, action="append")
    # parser.add_argument("--optim_s_list", type=int, action="append")
    # parser.add_argument("--optim_d_list", type=int, action="append")
    # parser.add_argument("--joint_training_step", type=int, default=60000)
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')

    parser.add_argument("--vis_train_sancheck", action="store_true")
    parser.add_argument("--vis_train_every", type=int, default=2000,
                        help='frequency of visualize the training on tensorboard')

    parser.add_argument("--vis_full_sancheck", action="store_true")
    parser.add_argument("--vis_full_every", type=int, default=10000,
                        help='frequency of visualize the full training')

    parser.add_argument("--save_sancheck", action="store_true")
    parser.add_argument("--save_every", type=int, default=20000)

    parser.add_argument("--multiGPU", type=int, default=None, action="append")
    parser.add_argument("--optimize_poses", type=int, default=0)
    parser.add_argument("--optimize_focal_length", type=int, default=0)
    parser.add_argument("--with_GT_poses", type=int, default=0)
    parser.add_argument("--multiview_dataset", type=int, default=0)
    parser.add_argument("--use_disp", type=int, default=0)
    # parser.add_argument("--use_foreground_mask", type=str, default="motion_masks", choices=["motion_masks", "epipolar_motion_masks", "epipolar_error_png"])
    parser.add_argument("--use_foreground_mask", type=str, default="motion_masks")
    parser.add_argument("--use_time_embedding", type=int, default=0)
    parser.add_argument("--time_embedding_size", type=int, default=4)
    parser.add_argument("--save_poses_bounds", type=int, default=0)

    parser.add_argument("--fov_init", type=float, default=30)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()