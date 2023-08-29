# stdlib imports
from datetime import date
import logging
import os
import random
from typing import List, Tuple, Union, Optional

# third-party imports
import matplotlib.pyplot as plt
from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator
import numpy as np
import plotly.graph_objects as go
from skimage.metrics import structural_similarity as ssim
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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
    model = M.NeRF(
            args.d_input,
            args.d_input,
            args.n_layers,
            args.d_filter, 
            args.skip,
            **kwargs
    )
    model.to(device) # move model to device
    
    return model

# TRAINING FUNCTIONS

def step(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[S] = None,
    verbose: bool = True,
    estimator: Optional[OccGridEstimator] = None,
    render_step_size: Optional[float] = None,
    mu: Optional[float] = None
    ) -> Tuple[float, float]:
    """
    Training/validation/test step.
    ----------------------------------------------------------------------------
    Reference(s):
        Based on 'step' function in train.py from NeRF repository by Yliess HATI. 
        Available at https://github.com/yliess86/NeRF
    ----------------------------------------------------------------------------
    Args:
        epoch (int): current epoch
        model (nn.Module): NeRF model
        loader (DataLoader): data loader
        device (torch.device): device to use for training
        split (str): split to use for training/validation/test
        optimizer (Optional[Optimizer]): optimizer to use for training
        scheduler (Optional[S]): scheduler to use for training
        testpose (Optional[Tensor]): test pose to use for visualization
        verbose (bool): whether to print progress bar
        estimator (Optional[OccGridEstimator]): occupancy grid estimator
        render_step_size (Optional[float]): step size for rendering
        mu (Optional[float]): weight for depth loss
    Returns:
        Tuple[float, float]: total loss, total PSNR
    ----------------------------------------------------------------------------
    """
    train = split == 'train'
    # set model to corresponding mode
    model = model.train(train)

    avg_mae, avg_psnr = 0., 0.
    
    # set up progress bar
    desc = f"[NeRF] {split.capitalize()} {epoch + 1}"
    batches = tqdm(loader, desc=desc, disable=not verbose)

    with torch.set_grad_enabled(train):
        for i, batch in enumerate(batches):
            step = epoch * len(loader) + i
            rays_o, rays_d, rgb_gt, depth_gt = batch
            
            # forward pass
            def occ_eval_fn(x):
                density = model(x)
                return density * render_step_size

            rgb, _, depth, _ = R.render_rays(
                    rays_o=rays_o,
                    rays_d=rays_d,
                    estimator=estimator,
                    device=device,
                    model=model,
                    train=train,
                    white_bkgd=args.white_bkgd,
                    render_step_size=render_step_size
            )

            # send g.t. data to device
            rgb_gt = rgb_gt.to(device)
            depth_gt = depth_gt.to(device)

            # compute loss and psnr
            loss = F.mse_loss(rgb, rgb_gt)
            with torch.no_grad():
                psnr = -10. * torch.log10(loss)

            # add depth loss if using depth supervision
            if mu is not None:
                depth_loss = L.depth_l1(
                        depth.squeeze(-1), 
                        depth_gt,
                        use_bkgd=args.use_bkgd
                )
                loss += mu * depth_loss

            with torch.no_grad():
                # remove bkgd if necessary
                depth = depth.squeeze(-1)
                mask = ~torch.isinf(depth_gt)
                mask = mask | ~mask if args.use_bkgd else mask
                depth = depth[mask]
                depth_gt = depth_gt[mask]
                # mean absolute error
                mae = torch.abs(depth - depth_gt)
                mae = torch.mean(mae)


            if train:
                # backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # update occupancy grid
                estimator.update_every_n_steps(
                    step=step,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                )

                if args.debug is False:
                    # log metrics to wandb
                    wandb.log({
                        'train_psnr': psnr.item(),
                        'train_depth_mae': mae.item(),
                        'lr': scheduler.lr
                    })

            # accumulate metrics
            avg_psnr += psnr.item() / len(loader)
            avg_mae += mae.item() / len(loader)

    return avg_psnr, avg_mae

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

    if args.debug is False:
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
    val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
    )

    # optimizer and scheduler
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lrate)
    scheduler = S.MipNerf(
            optimizer, 
            args.n_iters,
            warmup_steps=args.warmup_iters,
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

    # compute number of epochs
    epochs = args.n_iters // len(train_loader)

    # set up progress bar
    desc = f"[NeRF] Epoch"
    pbar = tqdm(range(int(epochs)), desc=desc)

    # initialize best validation metrics
    best_psnr = 0.
    best_mae = float('inf')
    # iterate over epochs
    for e in pbar:
        # training step
        _, _ = step(
                epoch=e, 
                model=model, 
                loader=train_loader, 
                device=device, 
                split='train', 
                optimizer=optimizer,
                scheduler=scheduler,
                estimator=estimator,
                render_step_size=render_step_size,
                mu=mu
        )
        # validation step
        val_psnr, val_mae = step(
                epoch=e,
                model=model,
                loader=val_loader,
                device=device,
                split='val',
                estimator=estimator,
                render_step_size=render_step_size,
                mu=mu
        )
        # update best validation metrics
        best_psnr = max(best_psnr, val_psnr)
        best_mae = min(best_mae, val_mae)

        if args.debug is False:
            # log validation metrics to wandb
            wandb.log({
                'val_psnr': val_psnr,
                'val_mae': val_mae,
            })
        # render test image
        with torch.no_grad():
            model.eval()
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
            if args.debug is False:
                # log images to wandb
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
    if args.debug is not True:
        wandb.login()
        # set up wandb run to track training
        wandb.init(
            project='depth-nerf',
            name='nerf' if args.mu is None else 'depth',
            config={
                'dataset': args.dataset,
                'scene': args.scene,
                'n_iters': args.n_iters,
                'lrate': args.lrate,
                'white_bkgd': args.white_bkgd,
                'use_bkgd': args.use_bkgd
            }
        )

    if args.sweep:
        mu = wandb.config.mu
    else:
        mu = args.mu
        
    n_imgs = args.n_imgs

    # build base path for output directories
    method = 'nerf' if args.mu is None else 'depth'
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
            white_bkgd=args.white_bkgd
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

    wandb.log({
        'camera_positions': fig
    })

    # load validation data
    val_set = D.SyntheticRealistic(
            scene=args.scene,
            n_imgs=n_imgs,
            split='val',
            white_bkgd=args.white_bkgd
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

    # render frames for all rendering poses
    output = R.render_path(
            render_poses=render_poses,
            hwf=[H, W, focal],
            chunksize=args.batch_size,
            device=device,
            model=model,
            train=False,
            white_bkgd=args.white_bkgd,
            render_step_size=render_step_size
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


# run in sweep mode or not
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
