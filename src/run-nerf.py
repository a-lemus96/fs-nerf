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
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
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

if args.debug is False:
    # set up wandb run to track training
    wandb.init(
        project='depth-nerf',
        name='nerf' if args.mu is None else 'depth',
        config={
            'dataset': args.dataset,
            'scene': args.scene,
            'use_fine': args.use_fine,
            'n_imgs': args.n_imgs,
            'n_iters': args.n_iters,
            'lrate': args.lrate,
            'mu': args.mu,
        }
    )

# bundle kwargs
kwargs_sample_stratified = {
    'n_samples': args.n_samples,
    'perturb': args.perturb,
    'inverse_depth': args.inv_depth
}

kwargs_sample_hierarchical = {
    'perturb': args.perturb_hierch
}

# use cuda device if available
cuda_available = torch.cuda.is_available()
device = torch.device(f'cuda:{args.device_num}' if cuda_available else 'cpu')

# print device info or abort if no CUDA device available
if device != 'cpu' :
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
else:
    raise RuntimeError("CUDA device not available.")

# build base path for output directories
method = 'nerf' if args.mu is None else 'depth'
out_dir = os.path.normpath(os.path.join(args.out_dir, method, 
                                        args.dataset, args.scene,
                                        'n_' + str(args.n_imgs),
                                        'lrate_' + str(args.lrate)))

# create folders
folders = ['video', 'model']
[os.makedirs(os.path.join(out_dir, f), exist_ok=True) for f in folders]

# load dataset(s)
train_set = D.SyntheticRealistic(
        scene=args.scene,
        split='train',
        white_bkgd=args.white_bkgd
)

val_set = D.SyntheticRealistic(
        scene=args.scene,
        split='val',
        white_bkgd=args.white_bkgd
)

# get near and far bounds, image dimensions and focal length
near, far = train_set.near, train_set.far
H, W, focal = train_set.hwf
H, W = int(H), int(W)

logger = logging.getLogger()
base_level = logger.level


# MODELS INITIALIZATION

def init_models():
    """
    Initialize models, encoders and optimizer for NeRF training
    """
    # encoders
    pos_encoder = M.PositionalEncoder(
            args.d_input, 
            args.n_freqs,
            log_space=args.log_space
    )
    pos_fn = lambda x: pos_encoder(x)

    # check if using view directions to initialize encoders
    if args.no_dirs is False:
        dir_encoder = M.PositionalEncoder(
                args.d_input, 
                args.n_freqs_views,
                log_space=args.log_space
        )
        dir_fn = lambda x: dir_encoder(x)
        d_viewdirs = dir_encoder.d_output
    else:
        dir_fn = None
        d_viewdirs = None

    # models
    coarse = M.NeRF(
            pos_encoder.d_output, 
            n_layers=args.n_layers,
            d_filter=args.d_filter, 
            skip=args.skip,
            d_viewdirs=d_viewdirs
    )
    coarse.to(device)
    params = list(coarse.parameters())
    if args.use_fine:
        fine = M.NeRF(
                pos_encoder.d_output, 
                n_layers=args.n_layers,
                d_filter=args.d_filter_fine, 
                skip=args.skip,
                d_viewdirs=d_viewdirs
        )
        fine.to(device)
        params = params + list(fine.parameters())
    else:
        fine = None
    
    return coarse, fine, params, pos_fn, dir_fn

# TRAINING FUNCTIONS

def step(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[S] = None,
    testpose: Optional[Tensor] = None,
    verbose: bool = True,
    estimator: Optional[OccGridEstimator] = None,
    render_step_size: Optional[float] = None,
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
    Returns:
        Tuple[float, float]: total loss, total PSNR
    ----------------------------------------------------------------------------
    """
    train = split == 'train'
    # set model to corresponding mode
    model = model.train(train)

    total_loss, total_psnr = 0., 0.
    
    # set up progress bar
    desc = f"[NeRF] {split.capitalize()} {epoch + 1}"
    batches = tqdm(loader, desc=desc, disable=not verbose)

    with torch.set_grad_enabled(train):
        for i, batch in enumerate(batches):
            step = e * steps_per_epoch + i
            rays_o, rays_d, rgb_gt, depth_gt = batch
            # send data to device
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            rgb_gt = rgb_gt.to(device)
            depth_gt = depth_gt.to(device)
            
            # forward pass
            def occ_eval_fn(x):
                x = pos_fn(x)  # apply positional encoding
                density = model(x)
                return density * render_step_size

            def sigma_fn(t_starts, t_ends, ray_indices):
                to = rays_o[ray_indices]
                td = rays_d[ray_indices]
                x = to + td * (t_starts + t_ends)[:, None] / 2.0
                x = pos_fn(x) # positional encoding
                sigmas = model(x)
                return sigmas.squeeze(-1)

            ray_indices, t_starts, t_ends = estimator.sampling(
                    rays_o, rays_d,
                    sigma_fn=sigma_fn,
                    render_step_size=render_step_size,
                    stratified=train
            )

            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                    to = rays_o[ray_indices]
                    td = rays_d[ray_indices]
                    x = to + td * (t_starts + t_ends)[:, None] / 2.0
                    x = pos_fn(x) # positional encoding
                    td = dir_fn(td) # pos encoding
                    out = model(x, td)
                    rgbs = out[..., :3]
                    sigmas = out[..., -1]
                    return rgbs, sigmas.squeeze(-1)

            render_bkgd = torch.tensor(
                    args.white_bkgd * torch.ones(3), 
                    device=device, 
                    requires_grad=True
            )

            rgb, opacity, depth, extras = rendering(
                    t_starts,
                    t_ends,
                    ray_indices,
                    n_rays=rays_o.shape[0],
                    rgb_sigma_fn=rgb_sigma_fn,
                    render_bkgd=render_bkgd
            )

            loss = F.mse_loss(rgb, rgb_gt) # compute loss
            with torch.no_grad():
                psnr = -10. * torch.log10(loss) # compute psnr

            # add depth loss if applicable
            if args.mu is not None:
                depth = outputs['depth']
                depth_loss = L.depth_l1(depth, depth_gt)
                loss += args.mu * depth_loss

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
                        'train_loss': loss.item(),
                        'lr': scheduler.lr
                    })

            if i % args.display_rate == 0:
                with torch.no_grad():
                    coarse.eval()
                    fine.eval() if fine is not None else None

                    # render test image
                    rgb, depth = U.render_frame(
                            H, W, focal, testpose,
                            args.batch_size, near, far,
                            pos_fn, coarse,
                            kwargs_sample_stratified=kwargs_sample_stratified,
                            n_samples_hierarchical=args.n_samples_hierch,
                            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                            fine_model=fine,
                            dir_fn=dir_fn,
                            white_bkgd=args.white_bkgd,
                            estimator=estimator,
                            render_step_size=render_step_size,
                            device=device
                    )

                    logger.setLevel(100)
                    logger.setLevel(base_level)

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
                    coarse = coarse.train(train)
                    fine = fine.train(train) if fine is not None else None

            # accumulate metrics
            total_loss += loss.item() / len(loader)
            total_psnr += psnr.item() / len(loader)

            # update progress bar
            batches.set_postfix(psnr=psnr.item())
            
    return total_loss, total_psnr

def train():
    r"""
    Run NeRF training loop.
    """
    testimg = train_set.testimg
    testdepth = train_set.testdepth
    testpose = train_set.testpose.to(device)

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
    steps_per_epoch = np.ceil(len(train_set)/args.batch_size)
    epochs = np.ceil(args.n_iters / steps_per_epoch)

    # set up progress bar
    desc = f"[NeRF] Epoch"
    pbar = tqdm(range(int(epochs)), desc=desc)

    for e in pbar:
        # train for one epoch
        train_loss, train_psnr = step(
                epoch=e, 
                model=coarse, 
                loader=train_loader, 
                device=device, 
                split='train', 
                optimizer=optimizer,
                scheduler=scheduler,
                testpose=testpose,
                estimator=estimator,
                render_step_size=render_step_size,
        )

        # validation after one epoch
        val_loss, val_psnr = step(
                epoch=e, 
                model=coarse, 
                loader=val_loader, 
                device=device, 
                split='val', 
                testpose=testpose,
                estimator=estimator,
                render_step_size=render_step_size
        )

        if args.debug is False:
            # log validation metrics to wandb
            wandb.log({
                'val_psnr': val_psnr,
                'val_loss': val_loss,
            })


if not args.render_only:
        coarse, fine, params, pos_fn, dir_fn = init_models()
        success, train_psnrs, val_psnrs, code = train()

        # save model
        torch.save(model.state_dict(), out_dir + '/model/nerf.pt')
        model.eval()
            
        if fine_model is not None:
            torch.save(fine_model.state_dict(),
                       out_dir + '/model/nerf_fine.pt')
            fine_model.eval()
else:
    model, fine_model, encode, params, encode_viewdirs = init_models()
    # load model
    model.load_state_dict(torch.load(out_dir + '/model/nerf.pt'))
    model.eval()
    if fine_model is not None:
        fine_model.load_state_dict(torch.load(out_dir + '/model/nerf_fine.pt'))
        fine_model.eval()

# compute path poses for rendering video output
render_poses = R.sphere_path()
render_poses = render_poses.to(device)

# Render frames for all rendering poses
output = R.render_path(render_poses=render_poses,
                       near=near,
                       far=far,
                       hwf=[H, W, focal],
                       chunksize=args.batch_size,
                       encode=encode,
                       model=model,
                       kwargs_sample_stratified=kwargs_sample_stratified,
                       n_samples_hierarchical=args.n_samples_hierch,
                       kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                       fine_model=fine_model,
                       encode_viewdirs=encode_viewdirs)

frames, d_frames = output

# Now we put together frames and save result into .mp4 file
R.render_video(basedir=f'{out_dir}/video/',
               frames=frames,
               d_frames=d_frames)
