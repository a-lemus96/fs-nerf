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

def init_model():
    """
    Initialize model for NeRF training
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

    model.to(device) # move model to device
    
    return model

# TRAINING FUNCTIONS

def train(
        model,
        train_set, 
        val_set,
        mu: Optional[float] = None
) -> Tuple[float, float]:
    """Train NeRF model.
    ----------------------------------------------------------------------------
    Args:
        model (nn.Module): NeRF model
        train_set (Dataset): training dataset
        val_set (Dataset): validation dataset
        mu (Optional[float]): weight for depth loss
    Returns:
        Tuple[float, float]: best validation PSNR, best validation MAE
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
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lrate)
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
                (args.lrate, 5e-4)
        )

    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    estimator = OccGridEstimator(
            roi_aabb=aabb, 
            resolution=grid_resolution, 
            levels=grid_nlvl
    ).to(device)

    # lpips network
    lpips_net = LPIPS(net='vgg').to(device)
    # initialize best validation metrics
    best_psnr = 0.
    best_mae = float('inf')

    pbar = tqdm(range(args.n_iters), desc=f"[NeRF]") # set up progress bar
    iterator = iter(train_loader)
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
            psnr = -10. * torch.log10(loss)
            # remove bkgd if necessary
            depth = depth.squeeze(-1)
            mask = ~torch.isinf(depth_gt)
            mask = mask | ~mask if args.use_bkgd else mask
            depth = depth[mask]
            depth_gt = depth_gt[mask]
            # mean absolute error
            mae = torch.abs(depth - depth_gt)
            mae = torch.mean(mae)

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
        if not args.debug:
            wandb.log({
                'train_psnr': psnr.item(),
                'train_depth_mae': mae.item(),
                'lr': scheduler.lr
            })

        # switch to validation mode
        mod = k % args.val_rate
        #if mod == 0 and k > 0:
        if mod == 0:
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
                val_lpips = lpips_net(rgbs, rgbs_gt).mean()
                rgbs = torch.permute(rgbs, (0, 2, 3, 1)).cpu().numpy()
                rgbs_gt = torch.permute(rgbs_gt, (0, 2, 3, 1)).cpu().numpy()
                val_ssim = np.mean(
                        SSIM(
                            rgbs, 
                            rgbs_gt, 
                            channel_axis=-1, 
                            gaussian_weights=True
                        )
                )
                # update best validation metrics
                best_psnr = max(best_psnr, val_psnr)

                # log validation metrics to wandb
                if not args.debug:
                    wandb.log({
                        'val_psnr': val_psnr,
                        'val_lpips': val_lpips,
                        'val_ssim': val_ssim
                    })

                # render test image
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
                # log images to wandb
                if not args.debug:
                    wandb.log({
                        'rgb': wandb.Image(
                            rgb.cpu().numpy(),
                            caption='RGB'
                        ),
                        'depth': wandb.Image(
                            PL.apply_colormap(depth.cpu().numpy()),
                            caption='Depth'
                        )
                    })

    return best_psnr, best_mae

def main():
    if not args.debug:
        wandb.login()
        # set up wandb run to track training
        wandb.init(
            project='depth-nerf',
            name='NeRF' if args.mu is None else 'FS',
            config={
                'dataset': args.dataset,
                'scene': args.scene,
                'img_mode': args.img_mode,
                'white_bkgd': args.white_bkgd,
                'model': args.model,
                'img_mode': args.img_mode,
                'scheduler': args.scheduler,
                'n_iters': args.n_iters,
                'n_imgs': args.n_imgs,
                'batch_size': args.batch_size,
                'lrate': args.lrate,
                'use_bkgd': args.use_bkgd,
                'val_ratio': args.val_ratio
            }
        )

    mu = args.mu
    n_imgs = args.n_imgs

    # build base path for output directories
    method = 'nerf' if args.mu is None else 'fs'
    out_dir = os.path.normpath(os.path.join(args.out_dir, method, 
                                            args.dataset, args.scene,
                                            'n_' + str(n_imgs),
                                            'lrate_' + str(args.lrate)))

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
        model = init_model() # initialize model
        # train model
        final_psnr, final_mae = train(
                model=model,
                train_set=train_set,
                val_set=val_set,
                mu=mu
        )
        wandb.log({
            'final_val_psnr': final_psnr,
            'final_val_mae': final_mae
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
            train=False,
            white_bkgd=args.white_bkgd,
            render_step_size=5e-3
    )

    frames, d_frames = output

    # Now we put together frames and save result into .mp4 file
    R.render_video(
            basedir=f'{out_dir}/video/',
            frames=frames,
            d_frames=d_frames
    )

# select device
cuda_available = torch.cuda.is_available()
device = torch.device(f'cuda:{args.device_num}' if cuda_available else 'cpu')

# print device info or abort if no CUDA device available
if device != 'cpu' :
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
else:
    raise RuntimeError("CUDA device not available.")


# either sweep or single training run
if not args.debug and args.sweep:
    # define sweep config
    sweep_config = {
        "program": "run-nerf.py",
        "method": "random",
        "metric": {
            "name": "final_val_psnr",
            "goal": "maximize"
        },
        "parameters": {
            "mu": {
                "values": [1e-8, 1e-7, 1e-6, 1e-5]
            },
            "beta": {
                "min": 0.02,
                "max": 0.5
            }
        }
    }
    # initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="depth-nerf"
    )
    # run sweep
    wandb.agent(sweep_id, function=main, count=10)
else:
    main()
