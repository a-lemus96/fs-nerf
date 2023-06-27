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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

# local imports
import core.models as M
import core.loss as L
import data.dataset as D 
import render.rendering as R
import utils.parser as P
import utils.utilities as U

# RANDOM SEED

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = P.config_parser() # parse command line arguments

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

# load dataset
dataset = D.SyntheticRealistic(
        scene=args.scene,
        n_imgs=args.n_imgs,
        white_bkgd=args.white_bkgd
    )
near, far = dataset.near, dataset.far
H, W, focal = dataset.hwf
H, W = int(H), int(W)

logger = logging.getLogger()
base_level = logger.level

# TRAINING CLASSES AND FUNCTIONS

def plot_samples(
    z_vals: torch.Tensor, 
    z_hierarch: Optional[torch.Tensor] = None,
    ax: Optional[np.ndarray] = None):
    r"""
    Plot stratified and (optional) hierarchical samples.
    Args:
    Returns:
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')

    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')
    
    ax.set_ylim([-1, 2])
    ax.set_title('Stratified Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)

    return ax

class EarlyStopping:
    r"""
    Early stopping helper class based on fitness criterion.
    """
    def __init__(
        self,
        patience: int = 30,
        margin: float = 1e-4
    ):
        self.best_fitness = 0.0 # PSNR measure
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float('inf') # epochs to wait after fitness
                                                 # stops improving to stop

    def __call__(
        self,
        iter: int,
        fitness: float
    ):
        r"""
        Check if stopping criterion is met.
        """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience # stop training if patience is exceeded

        return stop


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

# TRAINING LOOP

# Early stopping helper
warmup_stopper = EarlyStopping(patience=2000)

def train():
    r"""
    Run NeRF training loop.
    """
    # Create data loader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(params, lr=args.lrate)
    scheduler = U.CustomScheduler(optimizer, args.n_iters, 
                                  n_warmup=args.warmup_iters)

    train_psnrs = []
    val_psnrs = []
    iternums = []
    sigma_curves = []
    
    testimg = dataset.testimg.to(device)
    testpose = dataset.testpose.to(device)
    testdepth = dataset.testdepth.to(device)

    # Compute number of epochs
    steps_per_epoch = np.ceil(len(dataset)/args.batch_size)
    epochs = np.ceil(args.n_iters / steps_per_epoch)

    for i in range(int(epochs)):
        print(f"Epoch {i + 1}")
        model.train()

        for k, batch in enumerate(tqdm(dataloader)):
            # Compute step
            step = int(i * steps_per_epoch + k)

            # Unpack batch info
            rays_o, rays_d, rgb_gt, depth_gt = batch
            
            # forward pass
            outputs = U.nerf_forward(rays_o.to(device), rays_d.to(device),
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=args.n_samples_hierch,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           dir_fn=encode_viewdirs,
                           white_bkgd=args.white_bkgd)

            # check for numerical errors
            for key, val in outputs.items():
                if torch.isnan(val).any():
                    print(f"! [Numerical Alert] {key} contains NaN.")
                if torch.isinf(val).any():
                    print(f"! [Numerical Alert] {key} contains Inf.")

            rgb_gt = rgb_gt.to(device) # ground truth rgb to device
            rgb = outputs['rgb_map'] # predicted rgb
            loss = F.mse_loss(rgb, rgb_gt) # compute loss
            # compute PSNR
            with torch.no_grad():
                psnr = -10. * torch.log10(loss)
                train_psnrs.append(psnr.item())

            if args.use_fine: # add coarse loss
                loss += F.mse_loss(outputs['rgb_map_0'], rgb_gt) # add

            # add depth loss
            if args.mu is not None:
                depth = outputs['depth_map']
                weight = outputs['weights']
                z_vals = outputs['z_vals_combined']
                depth_gt = depth_gt.to(device)
                depth_loss = L.depth(depth, depth_gt, weight, z_vals)
                loss += args.mu * depth_loss

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % args.val_rate == 0:
                with torch.no_grad():
                    model.eval()

                    rays_o, rays_d = U.get_rays(H, W, focal, testpose)
                    rays_o = rays_o.reshape([-1, 3])
                    rays_d = rays_d.reshape([-1, 3])
                    
                    origins = U.get_chunks(rays_o, chunksize=args.batch_size)
                    dirs = U.get_chunks(rays_d, chunksize=args.batch_size)

                    rgb = []
                    depth = []
                    sigma = []
                    z_vals = []
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
                        sigma.append(outputs['sigma'])
                        z_vals.append(outputs['z_vals_combined'])

                    rgb = torch.cat(rgb, dim=0)
                    depth = torch.cat(depth, dim=0)
                    sigma = torch.cat(sigma, dim=0)
                    z_vals = torch.cat(z_vals, dim=0)

                    val_loss = F.mse_loss(rgb, testimg.reshape(-1, 3))
                    val_psnr = -10. * torch.log10(val_loss)

                    val_psnrs.append(val_psnr.item())
                    iternums.append(step)
                    if step % args.display_rate == 0:
                        # Save density distribution along sample ray
                        z_vals = z_vals.view(-1,
                                args.n_samples + args.n_samples_hierch)
                        #sample_idx = 320350
                        red_coords = (400, 400)
                        flatten_coords = lambda x, y: y * W + x
                        sample_idx = flatten_coords(*red_coords)
                        z_sample = z_vals[sample_idx].detach().cpu().numpy()
                        sigma_sample = sigma[sample_idx].detach().cpu().numpy()
                        curve = np.concatenate((z_sample[..., None],
                                                sigma_sample[..., None]), -1)
                        sigma_curves.append(curve)

                        logger.setLevel(100)

                        # Plot example outputs
                        fig, ax = plt.subplots(2, 3, figsize=(25, 8),
                                               gridspec_kw={'width_ratios': [1, 1, 3]})
                        ax[0,0].imshow(rgb.reshape([H, W, 3]).cpu().numpy())
                        ax[0,0].set_title(f'Iteration: {step}')
                        ax[0,1].imshow(testimg.cpu().numpy())
                        ax[0,1].set_title(f'G.T. RGB')
                        ax[0,2].plot(range(0, step + 1), train_psnrs, 'r')
                        ax[0,2].plot(iternums, val_psnrs, 'b')
                        ax[0,2].set_title('PSNR (train=red, val=blue)')
                        ax[1,0].plot(red_coords, marker='o', color="red")
                        ax[1,0].imshow(depth.reshape([H, W]).cpu().numpy())
                        ax[1,0].set_title(r'Predicted Depth')
                        ax[1,1].plot(red_coords, marker='o', color="red")
                        ax[1,1].imshow(testdepth.cpu().numpy())
                        ax[1,1].set_title(r'G.T. Depth')
                        ax[1, 2].plot(z_sample, sigma_sample)
                        ax[1, 2].set_title('Density along sample ray (red dot)')
                        plt.savefig(f"{out_dir}/training/iteration_{step}.png")
                        plt.close(fig)
                        logger.setLevel(base_level)

                        # Save density curves for sample ray
                        curves = np.array(sigma_curves)
                        np.savez(out_dir + '/training_info',
                                 t_psnrs=train_psnrs,
                                 v_psnrs=val_psnrs,
                                 curves=curves)

            # Check PSNR for issues and stop if any are found.
            if step == args.warmup_iters - 1:
                if val_psnr < args.min_fitness:
                    return False, train_psnrs, val_psnrs, 0
            elif step < args.warmup_iters:
                if warmup_stopper is not None and warmup_stopper(step, val_psnr):
                    return False, train_psnrs, val_psnrs, 1 

        print("Loss:", val_loss.item())

    return True, train_psnrs, val_psnrs, 2

if not args.render_only:
    # Run training session(s)
    for k in range(args.n_restarts):
        print('Training attempt: ', k + 1)
        model, fine_model, encode, params, encode_viewdirs = init_models()
        success, train_psnrs, val_psnrs, code = train()

        if success and val_psnrs[-1] >= args.min_fitness:
            print('Training successful!')

            # Save model
            torch.save(model.state_dict(), out_dir + '/model/nerf.pt')
            model.eval()
            
            if fine_model is not None:
                torch.save(fine_model.state_dict(),
                           out_dir + '/model/nerf_fine.pt')
                fine_model.eval()

            break
        if not success and code == 0:
            print(f'Val PSNR {val_psnrs[-1]} below warmup_min_fitness {args.min_fitness}. Stopping...')
        elif not success and code == 1:
            print(f'Train PSNR flatlined for {warmup_stopper.patience} iters. Stopping...')

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
