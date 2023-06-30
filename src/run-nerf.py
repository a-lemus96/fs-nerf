# stdlib imports
from datetime import date
import logging
import os
import random
from typing import List, Tuple, Union, Optional

# third-party imports
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

# local imports
import core.models as M
import core.loss as L
import data.dataset as D 
import render.rendering as R
import utils.parser as P
import utils.plotting as PL
import utils.utilities as U

# RANDOM SEED

#seed = 43
#torch.manual_seed(seed)
#np.random.seed(seed)
#random.seed(seed)

args = P.config_parser() # parse command line arguments

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

# Bundle the kwargs for various functions to pass all at once
kwargs_sample_stratified = {
    'n_samples': args.n_samples,
    'perturb': args.perturb,
    'inverse_depth': args.inv_depth
}

kwargs_sample_hierarchical = {
    'perturb': args.perturb_hierch
}

# Use cuda device if available
cuda_available = torch.cuda.is_available()
device = torch.device(f'cuda:{args.device_num}' if cuda_available else 'cpu')

# Verify CUDA availability
if device != 'cpu' :
    print(f"CUDA device: {torch.cuda.get_device_name(device)}\n")
else:
    raise RuntimeError("CUDA device not available.")
    exit()

# build base path for output directories
method = 'nerf' if args.mu is None else 'depth'
out_dir = os.path.normpath(os.path.join(args.out_dir, method, 
                                        args.dataset, args.scene,
                                        'n_' + str(args.n_imgs),
                                        'lrate_' + str(args.lrate)))

# create folders
folders = ['training', 'video', 'model']
[os.makedirs(os.path.join(out_dir, f), exist_ok=True) for f in folders]

# load dataset(s)
train_set = D.SyntheticRealistic(
        scene=args.scene,
        split='train',
        n_imgs=args.n_imgs,
        white_bkgd=args.white_bkgd
    )

val_set = D.SyntheticRealistic(
        scene=args.scene,
        split='val',
        n_imgs=args.n_imgs//2,
        white_bkgd=args.white_bkgd
    )

near, far = train_set.near, train_set.far
H, W, focal = train_set.hwf
H, W = int(H), int(W)

logger = logging.getLogger()
base_level = logger.level


# MODELS INITIALIZATION

def init_models():
    r"""
    Initialize models, encoders and optimizer for NeRF training
    """
    # Encoders
    encoder = M.PositionalEncoder(args.d_input, args.n_freqs,
                                  log_space=args.log_space)
    encode = lambda x: encoder(x)

    # Check if using view directions to initialize encoders
    if args.no_dirs is False:
        encoder_viewdirs = M.PositionalEncoder(args.d_input, args.n_freqs_views,
                                               log_space=args.log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = M.NeRF(encoder.d_output, n_layers=args.n_layers,
                   d_filter=args.d_filter, skip=args.skip,
                   d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if args.use_fine:
        fine_model = M.NeRF(encoder.d_output, n_layers=args.n_layers,
                            d_filter=args.d_filter_fine, skip=args.skip,
                            d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None
    
    return model, fine_model, encode, model_params, encode_viewdirs 

# TRAINING FUNCTIONS

def step(
    epoch: int,
    coarse: nn.Module,
    optimizer: Optimizer,
    scheduler: U.CustomScheduler,
    loader: DataLoader,
    device: torch.device,
    split: str,
    fine: Optional[nn.Module] = None,
    verbose: bool = True,
    ) -> Tuple[float, float, float]:
    """
    Training/validation/test step.
    ----------------------------------------------------------------------------
    Reference(s):
        Based on 'step' function in train.py from NeRF repository by Yliess HATI. 
        Available at https://github.com/yliess86/NeRF
    ----------------------------------------------------------------------------
    Args:
        epoch (int): Current epoch
        coarse (nn.Module): Coarse NeRF model
        optimizer (Optimizer): Optimizer
        scheduler (U.CustomScheduler): Learning rate scheduler
        loader (DataLoader): Data loader
        device (torch.device): Device to use for training
        split (str): Either 'train', 'val' or 'test'
        fine (Optional[nn.Module]): Fine NeRF model
        verbose (bool): Whether to display progress bar
    Returns:
        
    ----------------------------------------------------------------------------
    """
    train = split == 'train'
    # set model(s) to corresponding mode
    coarse = coarse.train(train)
    fine = fine.train(train) if fine is not None else None

    total_loss, total_psnr = 0., 0.
    
    # set up progress bar
    desc = f"[NeRF] {split.capitalize()} {epoch + 1}"
    batches = tqdm(loader, desc=desc, disable=not verbose)

    with torch.set_grad_enabled(train):
        for batch in batches:
            rays_d, rays_o, rgb_gt, depth_gt = batch
            # send data to device
            rays_d = rays_d.to(device)
            rays_o = rays_o.to(device)
            rgb_gt = rgb_gt.to(device)
            depth_gt = depth_gt.to(device)
            
            # forward pass
            outputs = U.nerf_forward(rays_o, rays_d,
                           near, far, encode, coarse,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=args.n_samples_hierch,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine,
                           dir_fn=encode_viewdirs,
                           white_bkgd=args.white_bkgd)

            # check for numerical errors
            for key, val in outputs.items():
                if torch.isnan(val).any():
                    print(f"[Numerical Alert] {key} contains NaN.")
                if torch.isinf(val).any():
                    print(f"[Numerical Alert] {key} contains Inf.")

            rgb = outputs['rgb_map'] # predicted rgb
            loss = F.mse_loss(rgb, rgb_gt) # compute loss
            # add coarse RGB loss 
            if fine is not None:
                rgb_coarse = outputs['rgb_map_0']
                loss += F.mse_loss(rgb_coarse, rgb_gt)

            with torch.no_grad():
                psnr = -10. * torch.log10(loss) # compute psnr

            # add depth loss if applicable
            if args.mu is not None:
                depth = outputs['depth_map']
                depth_loss = L.depth_l1(depth, depth_gt)
                loss += args.mu * depth_loss
                # add coarse depth loss 
                if fine is not None:
                    depth_coarse = outputs['depth_map_0']
                    loss += args.mu * L.depth_l1(depth_coarse, depth_gt)

            if train:
                # backward pass
                loss.backward()
                optimizer.zero_grad()
                scheduler.step()
                lr = scheduler.get_lr()
                # log metrics to wandb
                wandb.log({
                    'train_psnr': psnr.item(),
                    'train_loss': loss.item(),
                    'lr': lr
                    })

            # accumulate metrics
            total_loss += loss.item() / len(loader)
            total_psnr += psnr.item() / len(loader)

            # update progress bar
            batches.set_postfix(loss=loss.item(), psnr=psnr.item(), lr=lr)
            
    return total_loss, total_psnr, lr

def train():
    r"""
    Run NeRF training loop.
    """
    testimg = val_set.testimg
    testdepth = val_set.testdepth
    testpose = val_set.testpose.to(device)
    # log test maps to wandb
    wandb.log({
        'rgb_gt': wandb.Image(
            testimg.numpy(),
            caption='Ground Truth RGB'
        ),
        'depth_gt': wandb.Image(
            testdepth.numpy(),
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
    scheduler = U.CustomScheduler(optimizer, args.n_iters, 
                                  n_warmup=args.warmup_iters)

    # compute number of epochs
    steps_per_epoch = np.ceil(len(train_set)/args.batch_size)
    epochs = np.ceil(args.n_iters / steps_per_epoch)
    for e in range(int(epochs)):
        # train for one epoch
        train_loss, train_psnr, _ = step(
                epoch=e, 
                coarse=model, 
                optimizer=optimizer,
                scheduler=scheduler,
                loader=train_loader, 
                device=device, 
                split='train', 
                fine=fine_model
        )
        # validation after one epoch
        val_loss, val_psnr, _ = step(
                epoch=e, 
                coarse=model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                loader=val_loader, 
                device=device, 
                split='val', 
                fine=fine_model
        )

        # log validation metrics to wandb
        wandb.log({
            'val_psnr': val_psnr.item(),
            'val_loss': val_loss.item(),
            })

        # compute test image and test depth map
        rays_o, rays_d = U.get_rays(H, W, focal, testpose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])
        
        origins = U.get_chunks(rays_o, chunksize=args.batch_size)
        dirs = U.get_chunks(rays_d, chunksize=args.batch_size)

        rgb = []
        depth = []
        for batch_o, batch_d in zip(origins, dirs):
            outputs = U.nerf_forward(batch_o, batch_d,
                   near, far, encode, model,
                   kwargs_sample_stratified=kwargs_sample_stratified,
                   n_samples_hierarchical=args.n_samples_hierch,
                   kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                   fine_model=fine_model,
                   dir_fn=encode_viewdirs,
                   white_bkgd=args.white_bkgd)

            rgb.append(outputs['rgb_map'])
            depth.append(outputs['depth_map'])

            rgb = torch.cat(rgb, dim=0)
            depth = torch.cat(depth, dim=0)
            sigma = torch.cat(sigma, dim=0)
            z_vals = torch.cat(z_vals, dim=0)

            logger.setLevel(100)

            # log images to wandb
            wandb.log({
                'rgb': wandb.Image(
                    rgb.reshape(H, W, 3).cpu().numpy(),
                    caption='RGB'
                ),
                'depth': wandb.Image(
                    depth.reshape(H, W).cpu().numpy(),
                    caption='Depth'
                )
            })

            logger.setLevel(base_level)

if not args.render_only:
        model, fine_model, encode, params, encode_viewdirs = init_models()
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
