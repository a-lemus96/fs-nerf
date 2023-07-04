# stdlib imports
from datetime import date
import logging
import os
import random
from typing import List, Tuple, Union, Optional

# third-party imports
import matplotlib.pyplot as plt
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
        n_imgs=args.n_imgs,
        white_bkgd=args.white_bkgd
)

val_set = D.SyntheticRealistic(
        scene=args.scene,
        split='val',
        n_imgs=args.n_imgs//2,
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
    coarse: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: torch.device,
    split: str,
    testpose: Tensor,
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
        loader (DataLoader): Data loader
        device (torch.device): Device to use for training
        split (str): Either 'train', 'val' or 'test'
        testpose (Tensor): Test pose
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
        for i, batch in enumerate(batches):
            rays_d, rays_o, rgb_gt, depth_gt = batch
            # send data to device
            rays_d = rays_d.to(device)
            rays_o = rays_o.to(device)
            rgb_gt = rgb_gt.to(device)
            depth_gt = depth_gt.to(device)
            
            # forward pass
            outputs = U.nerf_forward(
                    rays_o, rays_d,
                    near, far, pos_fn, coarse,
                    kwargs_sample_stratified=kwargs_sample_stratified,
                    n_samples_hierarchical=args.n_samples_hierch,
                    kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                    fine=fine,
                    dir_fn=dir_fn,
                    white_bkgd=args.white_bkgd
            )

            if i % args.display_rate == 0:
                with torch.no_grad():
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
                            white_bkgd=args.white_bkgd
                    )

                    logger.setLevel(100)

                    if args.debug is False:
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

            # check for numerical errors
            for key, val in outputs.items():
                if torch.isnan(val).any():
                    print(f"[Numerical Alert] {key} contains NaN.")
                if torch.isinf(val).any():
                    print(f"[Numerical Alert] {key} contains Inf.")

            rgb = outputs['rgb'] # predicted rgb
            loss = F.mse_loss(rgb, rgb_gt) # compute loss

            with torch.no_grad():
                psnr = -10. * torch.log10(loss) # compute psnr

            # add coarse RGB loss 
            if fine is not None:
                rgb_coarse = outputs['rgb0']
                loss += F.mse_loss(rgb_coarse, rgb_gt)

            # add depth loss if applicable
            if args.mu is not None:
                depth = outputs['depth']
                depth_loss = L.depth_l1(depth, depth_gt)
                loss += args.mu * depth_loss
                # add coarse depth loss
                if fine is not None:
                    depth_coarse = outputs['depth0']
                    loss += args.mu * L.depth_l1(depth_coarse, depth_gt)

            if train:
                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if args.debug is False:
                    # log metrics to wandb
                    wandb.log({
                        'train_psnr': psnr.item(),
                        'train_loss': loss.item(),
                    })

            # accumulate metrics
            total_loss += loss.item() / len(loader)
            total_psnr += psnr.item() / len(loader)

            # update progress bar
            batches.set_postfix(loss=loss.item(), psnr=psnr.item())
            
    return total_loss, total_psnr

def train():
    r"""
    Run NeRF training loop.
    """
    testimg = val_set.testimg
    testdepth = val_set.testdepth
    testpose = val_set.testpose.to(device)
    if args.debug is False:
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

    # compute number of epochs
    steps_per_epoch = np.ceil(len(train_set)/args.batch_size)
    epochs = np.ceil(args.n_iters / steps_per_epoch)
    for e in range(int(epochs)):
        # train for one epoch
        train_loss, train_psnr = step(
                epoch=e, 
                coarse=coarse, 
                optimizer=optimizer,
                loader=train_loader, 
                device=device, 
                split='train', 
                testpose=testpose,
                fine=fine
        )
        # validation after one epoch
        val_loss, val_psnr = step(
                epoch=e, 
                coarse=coarse, 
                optimizer=optimizer, 
                loader=val_loader, 
                device=device, 
                split='val', 
                fine=fine
        )

        if args.debug is False:
            # log validation metrics to wandb
            wandb.log({
                'val_psnr': val_psnr.item(),
                'val_loss': val_loss.item(),
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
