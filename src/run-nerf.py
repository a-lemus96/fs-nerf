# stdlib imports
import argparse
from datetime import date
import logging
import os
import random
from typing import List, Tuple, Union, Optional

# third-party imports
from multiprocessing import cpu_count
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# local imports
import data.dataset as D 
import core.models as M
import render.rendering as R
import utils.utilities as U

# RANDOM SEED

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# HYPERPARAMETERS 

parser = argparse.ArgumentParser(description='Train NeRF for view synthesis.')

# Encoder(s)
parser.add_argument('--d_input', dest='d_input', default=3, type=int,
                    help='Spatial input dimension')
parser.add_argument('--n_freqs', dest='n_freqs', default=10, type=int,
                    help='Number of encoding functions for spatial coords')
parser.add_argument('--log_space', dest='log_space', action="store_false",
                    help='If not set, frecuency scale in log space')
parser.add_argument('--no_dirs', dest='no_dirs', action="store_true",
                    help='If set, do not model view-dependent effects')
parser.add_argument('--n_freqs_views', dest='n_freqs_views', default=4,
                    type=int, help='Number of encoding functions for view dirs')

# Model(s)
parser.add_argument('--d_filter', dest='d_filter', default=256, type=int,
                    help='Linear layer filter dimension')
parser.add_argument('--n_layers', dest='n_layers', default=8, type=int,
                    help='Number of layers preceding bottleneck')
parser.add_argument('--skip', dest='skip', default=[4], type=list,
                    help='Layers at which to apply input residual')
parser.add_argument('--use_fine', dest='use_fine', action="store_true",
                    help='Creates and uses fine NeRF model')
parser.add_argument('--d_filter_fine', dest='d_filter_fine', default=256,
                    type=int, help='Linear layer filter dim for fine model')
parser.add_argument('--n_layers_fine', dest='n_layers_fine', default=8,
                    type=int, help='Number of fine layers preceding bottleneck')

# Stratified sampling
parser.add_argument('--n_samples', dest='n_samples', default=64, type=int,
                    help='Number of stratified samples per ray')
parser.add_argument('--perturb', dest='perturb', action="store_false",
                    help='If set, do not apply noise to spatial coords')
parser.add_argument('--inv_depth', dest='inv_depth', action="store_true",
                    help='If set, sample points linearly in inverse depth')

# Hierarchical sampling
parser.add_argument('--n_samples_hierch', dest='n_samples_hierch', default=128,
                    type=int, help='Number of hierarchical samples per ray')
parser.add_argument('--perturb_hierch', dest='perturb_hierch', action="store_false", 
                    help='Applies noise to hierarchical samples')

# Dataset
parser.add_argument('--dataset', dest='dataset', default='synthetic', type=str,
                    help="Dataset to choose scenes from")
parser.add_argument('--scene', dest='scene', default='lego', type=str,
                    help="Scene to be used for training")
parser.add_argument('--white_bkgd', dest='white_bkgd', default=False,
                    type=bool, help="Use white backgroung for training imgs")

# Optimization
parser.add_argument('--lrate', dest='lrate', default=5e-4, type=float,
                    help='Learning rate')

# Training 
parser.add_argument('--n_iters', dest='n_iters', default=10**5, type=int,
                    help='Number of training iterations')
parser.add_argument('--batch_size', dest='batch_size', default=2**12, type=int,
                    help='Number of rays per optimization step')
parser.add_argument('--device_num', dest='device_num', default=1, type=int,
                    help="Number of CUDA device to be used for training")

# Validation
parser.add_argument('--display_rate', dest='display_rate', default=1000, type=int,
                    help='Display rate for test output measured in iterations')
parser.add_argument('--val_rate', dest='val_rate', default=100, type=int,
                    help='Test image evaluation rate')

# Early Stopping
parser.add_argument('--warmup_iters', dest='warmup_iters', default=1e3,
                    type=int, help='Number of iterations for warmup phase')
parser.add_argument('--min_fitness', dest='min_fitness', default=10.0,
                    type=float, help='Minimum PSNR value to continue training')
parser.add_argument('--n_restarts', dest='n_restarts', default=5, type=int,
                    help='Maximum number of restarts if training stalls')

# Directories
parser.add_argument('--out_dir', dest='out_dir', default="../out/",
                    type=str, help="Base directory for storing results")

# Video Rendering
parser.add_argument('--render_only', dest='render_only', action='store_true',
                    help='If set, load pretrained model to render video')

args = parser.parse_args()

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

# Build base path for output directories
out_dir = os.path.normpath(os.path.join(args.out_dir, 'nerf', 
                                        args.dataset, args.scene,
                                        'viewdirs_' + str(not args.no_dirs),
                                        'lrate_' + str(args.lrate)))

# Create folders
folders = ['training', 'video', 'model']
[os.makedirs(os.path.join(out_dir, f), exist_ok=True) for f in folders]

# Load dataset
dataset = D.SyntheticRealistic(args.scene, 'test', white_bkgd=args.white_bkgd)
near, far = dataset.near, dataset.far
H, W, focal = dataset.hwf
H, W = int(H), int(W)

testimg, testpose = dataset.testimg, dataset.testpose

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
    scheduler = CustomScheduler(optimizer, args.n_iters, 
                                n_warmup=args.warmup_iters)

    train_psnrs = []
    val_psnrs = []
    iternums = []
    sigma_curves = []
    
    testimg, testpose = dataset.testimg.to(device), dataset.testpose.to(device)

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
            rays_o, rays_d, rgb_gt = batch
            
            # forward pass
            outputs = nerf_forward(rays_o.to(device), rays_d.to(device),
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

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if step % args.val_rate == 0:
                with torch.no_grad():
                    model.eval()

                    rays_o, rays_d = get_rays(H, W, focal, testpose)
                    rays_o = rays_o.reshape([-1, 3])
                    rays_d = rays_d.reshape([-1, 3])
                    
                    origins = get_chunks(rays_o, chunksize=args.batch_size)
                    dirs = get_chunks(rays_d, chunksize=args.batch_size)

                    rgb = []
                    depth = []
                    sigma = []
                    z_vals = []
                    for batch_o, batch_d in zip(origins, dirs):
                        outputs = nerf_forward(batch_o, batch_d,
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
                        sample_idx = 65010
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
                        ax[0,1].set_title(f'Target')
                        ax[0,2].plot(range(0, step + 1), train_psnrs, 'r')
                        ax[0,2].plot(iternums, val_psnrs, 'b')
                        ax[0,2].set_title('PSNR (train=red, val=blue')
                        #ax[1,0].plot(300, 400, marker='o', color="red")
                        ax[1,0].plot(210, 150, marker='o', color="red")
                        ax[1,0].imshow(depth.reshape([H, W]).cpu().numpy())
                        ax[1,0].set_title(r'Predicted Depth')
                        #ax[1,1].plot(300, 400, marker='o', color="red")
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
