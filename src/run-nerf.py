# stdlib imports
from datetime import date
import logging
import os
import random
from typing import List, Tuple, Union, Optional

# third-party imports
from lpips import LPIPS
import matplotlib.pyplot as plt
from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator
import numpy as np
import plotly.graph_objects as go
from skimage.metrics import structural_similarity as SSIM
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
from torch import Tensor
from tqdm import tqdm
import wandb

# local imports
import core.models as M
import core.loss as L
import core.scheduler as S
import data.dataset as D
import render.rendering as R
import utils.parser as P
import utils.plotting as PL
import utils.utilities as U

# GLOBAL VARIABLES
k = 0 # global step counter

# RANDOM SEED
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = P.config_parser() # parse command line arguments

# MODEL INITIALIZATION

def init_model() -> Tuple[nn.Module, OccGridEstimator]:
    """
    Initialize NeRF-like model and occupancy grid estimator.
    ----------------------------------------------------------------------------
    Returns:
        Tuple[nn.Module, OccGridEstimator]: NeRF model, OccGrid estimator
    """
    # keyword args for positional encoding
    kwargs = {
            'pos_fn': {
                'n_freqs': args.n_freqs,
                'log_space': args.log_space
            },
            'dir_fn': {
                'n_freqs': args.n_freqs_views,
                'log_space': args.log_space
            }
    }

    # instantiate model
    if args.model == 'nerf':
        model = M.NeRF(
                args.d_input,
                args.d_input,
                args.n_layers,
                args.d_filter, 
                args.skip,
                **kwargs
        )
    elif args.model == 'sinerf':
        model = M.SiNeRF(
                args.d_input,
                args.d_input,
                args.d_filter,
                [35., 1., 1., 1., 1., 1., 1., 1.]
        )

    # initialize occupancy estimator
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    estimator = OccGridEstimator(
            roi_aabb=aabb, 
            resolution=grid_resolution, 
            levels=grid_nlvl
    )
    
    return model, estimator

# TRAINING FUNCTIONS

def validation(
        hwf: Tensor,
        model: nn.Module,
        lpips_net: LPIPS,
        estimator: OccGridEstimator,
        val_loader: DataLoader,
        device: torch.device,
        render_step_size: float = 5e-3,
        val_samples: int = 10,
) -> Tuple[float, float, float]:
    """
    Perform validation step using a subset of the validation set.
    ----------------------------------------------------------------------------
    Args:
        model (nn.Module): NeRF-like model
        lpips_net (LPIPS): LPIPS network
        estimator (OccGridEstimator): occupancy grid estimator
        val_loader (DataLoader): validation set loader
        device (torch.device): device to train on
        render_step_size (float, optional): step size for rendering
        val_samples (int, optional): number of samples from val set to use
    Returns:
        val_psnr (float): validation PSNR
        val_ssim (float): validation SSIM
        val_lpips (float): validation LPIPS
    """
    H, W, focal = hwf
    H, W = int(H), int(W)
    model.eval()
    with torch.no_grad():
        val_iterator = iter(val_loader)
        rgbs = []
        rgbs_gt = []
        for v in range(val_samples):
            rgb_gt, depth_gt, pose = next(val_iterator)
            rgbs_gt.append(rgb_gt) # append ground truth rgb
            rgb, depth = R.render_frame(
                    H, W, focal, pose[0],
                    args.batch_size,
                    estimator,
                    device,
                    model,
                    train=False,
                    white_bkgd=args.white_bkgd,
                    render_step_size=render_step_size
            )
            rgbs.append(rgb) # append rendered rgb
        # compute validation metrics
        rgbs = torch.permute(torch.stack(rgbs, dim=0), (0, 3, 1, 2))
        rgbs_gt = torch.permute(torch.cat(rgbs_gt, dim=0), (0, 3, 1, 2))
        rgbs_gt = rgbs_gt.to(device)
        val_psnr = -10. * torch.log10(F.mse_loss(rgbs, rgbs_gt))
        rgbs = torch.permute(rgbs, (0, 2, 3, 1)).cpu().numpy()
        rgbs_gt = torch.permute(rgbs_gt, (0, 2, 3, 1)).cpu().numpy()
        ssims = np.zeros((rgbs.shape[0],))
        for i, (rgb, rgb_gt) in enumerate(zip(rgbs, rgbs_gt)):
            ssims[i] = SSIM(
                    rgb, 
                    rgb_gt, 
                    channel_axis=-1, 
                    data_range=1.,
                    gaussian_weights=True
            )
        val_ssim = np.mean(ssims)
        try:
            val_lpips = lpips_net(rgbs, rgbs_gt).mean()
        except:
            val_lpips = 1.

        return val_psnr, val_ssim, val_lpips

def train(
        model: nn.Module,
        estimator: OccGridEstimator,
        train_set: Dataset,
        val_set: Dataset,
        device: torch.device,
        render_step_size: float = 5e-3,
) -> Tuple[float, float]:
    """Train NeRF model.
    ----------------------------------------------------------------------------
    Args:
        model (nn.Module): NeRF model
        estimator (OccGridEstimator): occupancy grid estimator
        train_set (Dataset): training dataset
        device (torch.device): device to train on
        val_set (Dataset): validation dataset
    Returns:
        Tuple[float, float, float]: validation PSNR, SSIM, LPIPS
    ----------------------------------------------------------------------------
    """
    # retrieve camera intrinsics
    H, W, focal = train_set.hwf
    H, W = int(H), int(W)

    testimg = train_set.testimg
    testdepth = train_set.testdepth
    bkgd = torch.isinf(testdepth)
    testdepth[bkgd] = 0.
    testpose = train_set.testpose

    if not args.debug:
        # log test maps to wandb
        wandb.log({
            'rgb_gt': wandb.Image(
                testimg.numpy(),
                caption='Ground Truth RGB'
            ),
            'depth_gt': wandb.Image(
                PL.apply_colormap(testdepth.numpy()),
                caption='Ground Truth Depth'
            )
        })

    # data loader(s)
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
    )
    val_samples = int(args.val_ratio * len(val_set)) # % of val samples
    val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=True,
            num_workers=8
    )

    # optimizer and scheduler
    lro, lrf = args.lro, args.lrf
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lro)
    if args.scheduler == 'mip':
        scheduler = S.MipNerf(
                optimizer, 
                args.n_iters,
                warmup_steps=args.warmup_iters,
        )
    elif args.scheduler == 'exp':
        scheduler = S.ExponentialDecay(
                optimizer,
                args.n_iters,
                (lro, lrf)
        )
    elif args.scheduler == 'proot':
        scheduler = S.RootP(
                optimizer,
                args.T_lr,
                (lro, lrf),
                p=args.p
        )

    # lpips network
    lpips_net = LPIPS(net='vgg').to(device)

    pbar = tqdm(range(args.n_iters), desc=f"[NeRF]") # set up progress bar
    iterator = iter(train_loader) # data iterator
    alpha = 0.
    for k in pbar: # loop over the number of iterations
        # get next batch of data
        try:
            rays_o, rays_d, rgb_gt, depth_gt = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            rays_o, rays_d, rgb_gt, depth_gt = next(iterator)

        # render rays
        model.train()
        rgb, _, depth, _ = R.render_rays(
                rays_o=rays_o,
                rays_d=rays_d,
                estimator=estimator,
                device=device,
                model=model,
                train=True,
                white_bkgd=args.white_bkgd,
                render_step_size=render_step_size
        )
        # compute loss, PSNR, and MAE
        rgb_gt = rgb_gt.to(device)
        depth_gt = depth_gt.to(device)
        loss = F.mse_loss(rgb, rgb_gt)
        with torch.no_grad():
            psnr = -10. * torch.log10(loss).item()
            # remove bkgd if necessary
            depth = depth.squeeze(-1)
            mask = ~torch.isinf(depth_gt)
            mask = mask | ~mask if args.use_bkgd else mask
            depth = depth[mask]
            depth_gt = depth_gt[mask]
            # mean absolute error
            mae = torch.abs(depth - depth_gt)
            mae = torch.mean(mae).item()

        # weight decay regularization
        if args.ao is not None:
            freq_reg = torch.tensor(0.).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name and param.shape[0] > 3:
                    if args.reg == 'l1':
                        freq_reg += torch.abs(param).sum()
                    else:
                        freq_reg += torch.square(param).sum().sqrt()

            a = ((1. - 0.5**args.p)/args.T * k + 0.5**args.p) ** (1./args.p)
            alpha = 2 * args.ao * (1. - min(1., a))
            loss += alpha * freq_reg

        # backpropagate loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # define occupancy evaluation function
        def occ_eval_fn(x):
            density = model(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
                step=k,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2
        )

        # log metrics
        if not args.debug and k % args.val_rate != 0:
            wandb.log({
                'train_psnr': psnr,
                'train_depth_mae': mae,
                'lr': scheduler.lr,
                'alpha': alpha
            })

        # compute validation
        if k % args.val_rate == 0:
            val_data = validation(
                    train_set.hwf,
                    model,
                    lpips_net,
                    estimator,
                    val_loader,
                    device,
                    val_samples=val_samples
            )
            val_psnr, val_ssim, val_lpips = val_data
            # render test image
            with torch.no_grad():
                rgb, depth = R.render_frame(
                        H, W, focal, testpose,
                        args.batch_size,
                        estimator,
                        device,
                        model,
                        train=False,
                        white_bkgd=args.white_bkgd,
                        render_step_size=render_step_size
                )
            # log data to wandb
            if not args.debug:
                wandb.log({
                    'train_psnr': psnr,
                    'train_depth_mae': mae,
                    'lr': scheduler.lr,
                    'alpha': alpha,
                    'val_psnr': val_psnr,
                    'val_ssim': val_ssim,
                    'val_lpips': val_lpips,
                    'rgb': wandb.Image(
                        rgb.cpu().numpy(),
                        caption='RGB'
                    ),
                    'depth': wandb.Image(
                        PL.apply_colormap(depth.cpu().numpy()),
                        caption='Depth'
                    )
                })

    # compute final validation
    val_data = validation(
            train_set.hwf,
            model,
            lpips_net,
            estimator,
            val_loader,
            device,
            val_samples=len(val_loader)
    )

    return val_data

def main():
    # select device
    cuda_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device_num}' if cuda_available else 'cpu')

    # print device info or abort if no CUDA device available
    if device != 'cpu' :
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        raise RuntimeError("CUDA device not available.")

    if not args.debug:
        wandb.login()
        # set up wandb run to track training
        name = f"{args.model}"
        name = name + f"-{args.reg}" if args.ao is not None else name
        name = name + f"-p={args.p}" if args.ao is not None else name
        name = name + f"-ao={args.ao:.2e}" if args.ao is not None else name
        wandb.init(
            project='depth-nerf',
            name=name,
            config=args
        )
    # build base path for output directories
    n_imgs = args.n_imgs
    method = 'nerf' if args.ao is None else 'fs'
    out_dir = os.path.normpath(
            os.path.join(
                args.out_dir, 
                method, 
                args.dataset,
                args.scene,
                f"n_{str(n_imgs)}",
                f"lrates_{str(args.lro)}_{str(args.lrf)}"
            )
    )

    # create output directories
    folders = ['video', 'model']
    [os.makedirs(os.path.join(out_dir, f), exist_ok=True) for f in folders]

    # load training data
    train_set = D.SyntheticRealistic(
            scene=args.scene,
            n_imgs=n_imgs,
            split='train',
            white_bkgd=args.white_bkgd,
            img_mode=args.img_mode
    )
    # log interactive 3D plot of camera positions
    fig = go.Figure(
            data=[go.Scatter3d(
                x=train_set.poses[:, 0, 3],
                y=train_set.poses[:, 1, 3],
                z=train_set.poses[:, 2, 3],
                mode='markers',
                marker=dict(size=7, opacity=0.8, color='red'),
            )],
            layout=go.Layout(
                margin=dict(l=20,r=20, t=20, b=20),
            )
    )
    # set fixed axis scales
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-5, 5]),
            yaxis=dict(range=[-5, 5]),
            zaxis=dict(range=[0, 5]),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    if not args.debug:
        wandb.log({
            'camera_positions': fig
        })

    # load validation data
    val_set = D.SyntheticRealistic(
            scene=args.scene,
            n_imgs=25,
            split='val',
            white_bkgd=args.white_bkgd,
            img_mode=True
    )

    if not args.render_only:
        model, estimator = init_model() # initialize model
        model.to(device)
        estimator.to(device)
        # train model
        final_psnr, final_ssim, final_lpips = train(
                model=model,
                estimator=estimator,
                train_set=train_set,
                val_set=val_set,
                device=device
        )
        # log final metrics to wandb
        if not args.debug:
            wandb.log({
                'final_psnr': val_psnr,
                'final_ssim': val_ssim,
                'final_lpips': val_lpips
            })
        # save model
        torch.save(model.state_dict(), out_dir + '/model/nerf.pt')
    else:
        model = init_model()
        # load model
        model.load_state_dict(torch.load(out_dir + '/model/nerf.pt'))

    model.eval()

    # compute path poses for rendering video output
    render_poses = R.sphere_path()
    render_poses = render_poses.to(device)

    H, W, focal = train_set.hwf
    H, W = int(H), int(W)

    # render frames for all rendering poses
    output = R.render_path(
            render_poses=render_poses,
            hwf=[H, W, focal],
            chunksize=args.batch_size,
            device=device,
            model=model,
            estimator=estimator,
            white_bkgd=args.white_bkgd
    )

    frames, d_frames = output

    # Now we put together frames and save result into .mp4 file
    R.render_video(
            basedir=f'{out_dir}/video/',
            frames=frames,
            d_frames=d_frames
    )

if __name__ == '__main__':
    main()
