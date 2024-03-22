# stdlib imports
from datetime import date
import logging
import os
import random
from typing import List, Tuple, Union, Optional

# third-party imports
from lpips import LPIPS
import matplotlib.pyplot as plt
import nerfacc
from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator
import numpy as np
import plotly.graph_objects as go
from skimage.metrics import structural_similarity as SSIM
import torch
import pdb
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
import render.renderer as R
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

def init_models(dataset) -> Tuple[nn.Module, R.Renderer]:
    """
    Instantiate radiance field model and its associated occgrid-based renderer.
    Additionally, instantiate LPIPS network for perception metrics.
    ----------------------------------------------------------------------------
    Args:
        dataset: Dataset ADT.
    Returns:
        Tuple[nn.Module, R.Renderer, LPIPS]: model, renderer and LPIPS network
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
    match args.model:
        case 'nerf':
            model = M.NeRF(
                    args.d_input,
                    args.d_input,
                    args.n_layers,
                    args.d_filter, 
                    args.skip,
                    **kwargs
            )
        case 'sinerf':
            model = M.SiNeRF(
                    args.d_input,
                    args.d_input,
                    args.d_filter,
                    [30., 1., 1., 1., 1., 1., 1., 1.],
            )
        case _:
            raise ValueError(f"Model {args.model} not supported")

    # instantiate OccGrid-based renderer
    near = dataset.near
    far = dataset.far
    chunksize = args.batch_size
    white_bkgd = args.white_bkgd
    kwargs = {'render_step_size': 5e-3,
              'aabb': dataset.aabb,
              'resolution': 128,
              'grid_nlevels': 1 if args.dataset == 'synthetic' else 4,
              'occ_thre': 1e-2}

    renderer = R.Renderer(near, far, chunksize, white_bkgd, **kwargs)

    # instantiate LPIPS network
    lpips_net = LPIPS(net='vgg')
    
    return model, renderer, lpips_net

# TRAINING FUNCTIONS

def validation(
        hwf: Tuple[int, int, float],
        model: nn.Module,
        renderer: R.Renderer,
        lpips_net: LPIPS,
        val_loader: DataLoader,
        device: torch.device,
) -> Tuple[float, float, float]:
    """
    Performs validation step for NeRF-like model.
    ----------------------------------------------------------------------------
    Args:
        hwf (3,): Camera intinsics
        model (nn.Module): NeRF-like model
        renderer (R.Renderer): OccGrid-based renderer
        lpips_net (LPIPS): LPIPS network
        val_loader (DataLoader): data loader
        device (torch.device): device to be used
    Returns:
        val_psnr (float): validation PSNR
        val_ssim (float): validation SSIM
        val_lpips (float): validation LPIPS
    """
    rgbs_gt, poses = next(iter(val_loader))
    ndc = val_loader.dataset.ndc
    
    renderer.set_chunksize(2*args.batch_size)
    pdb.set_trace()
    rgbs, _ = renderer.render_poses(hwf,
                                    poses,
                                    model,
                                    ndc,
                                    device)
    renderer.set_chunksize(args.batch_size)

    # compute PSNR
    rgbs = torch.permute(rgbs, (0, 3, 1, 2))
    rgbs_gt = torch.permute(rgbs_gt, (0, 3, 1, 2))
    rgbs_gt = rgbs_gt.to(device)
    val_psnr = -10. * torch.log10(F.mse_loss(rgbs, rgbs_gt))
    val_size = len(val_loader.dataset)

    # compute LPIPS
    '''if val_size < 25:
        val_lpips = lpips_net(rgbs, rgbs_gt).mean()
    else:
        # compute LPIPS in chunks
        n_chunks = 5
        chunk_size = val_size //  n_chunks
        chunk_idxs = [i for i in range(0, val_size, chunk_size)]
        chunks = [(rgbs[i:i+chunk_size], rgbs_gt[i:i+chunk_size]) 
                  for i in chunk_idxs]
        val_lpips = 0.
        for chunk, chunk_gt in chunks:
            val_lpips += lpips_net(chunk, chunk_gt).mean()
        val_lpips /= n_chunks'''
    val_lpips = None

    # compute SSIM
    rgbs = torch.permute(rgbs, (0, 2, 3, 1)).cpu().numpy()
    rgbs_gt = torch.permute(rgbs_gt, (0, 2, 3, 1)).cpu().numpy()
    ssims = np.zeros((rgbs.shape[0],))
    val_ssim = 0.
    for rgb, rgb_gt in zip(rgbs, rgbs_gt):
        val_ssim += SSIM(
                rgb, 
                rgb_gt, 
                channel_axis=-1, 
                data_range=1.,
                gaussian_weights=True
        )
    val_ssim /= len(rgbs)

    return val_psnr, val_ssim, val_lpips

def train(
        model: nn.Module,
        renderer: R.Renderer,
        lpips_net: LPIPS,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = torch.device('cpu')
) -> Tuple[float, float]:
    """Train NeRF model.
    ----------------------------------------------------------------------------
    Args:
        model (nn.Module): NeRF model
        renderer (R.Renderer): OccGrid-based renderer
        lpips_net (LPIPS): LPIPS network
        train_loader (DataLoader): training set loader
        val_loader (DataLoader): validation set loader
        device (torch.device): device to train on
    ----------------------------------------------------------------------------
    """
    # retrieve camera intrinsics
    hwf = train_loader.dataset.hwf
    H, W, focal = hwf
    testpose = train_loader.dataset.testpose
    ndc = train_loader.dataset.ndc

    # set up optimizer and scheduler
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lro)
    sc_dict = {
            'const': (S.Constant, {}),
            'exp': (S.ExponentialDecay, {'r': args.decay_rate})
    }
    class_name, kwargs = sc_dict[args.scheduler]
    scheduler = class_name(
            optimizer,
            args.n_iters,
            args.lro,
            **kwargs
    )
    pbar = tqdm(range(args.n_iters), desc=f"[NeRF]") # set up progress bar
    iterator = iter(train_loader) # data iterator

    # regularizers
    if args.beta is not None:
        occ_reg = L.OcclusionRegularizer(args.beta, args.M)

    alpha = args.ao
    for k in pbar: # loop over the number of iterations
        model.train()
        renderer.train()
        # get next batch of data
        try:
            rays_o, rays_d, rgb_gt = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            rays_o, rays_d, rgb_gt = next(iterator)

        # render rays
        rays_o, rays_d = rays_o.to(device), rays_d.to(device)
        render_output = renderer.render_rays(rays_o, rays_d, model)
        rgb, _, depth, _ = render_output    # unpack rendering output
        
        # compute loss and PSNR
        rgb_gt = rgb_gt.to(device)
        loss = F.mse_loss(rgb, rgb_gt)
        with torch.no_grad():
            psnr = -10. * torch.log10(loss).item()

        '''# occlusion regularization
        if args.beta is not None:
            sigmas = extras['sigmas']
            if len(sigmas) > 0:
                loss += occ_reg(sigmas, ray_indices)'''

        # weight decay regularization
        if alpha is not None:
            freq_reg = torch.tensor(0.).to(device)
            # linear decay schedule
            Ts = int(args.reg_ratio * args.Td)
            if k < Ts:
                for name, param in model.named_parameters():
                    if 'weight' in name and param.shape[0] > 3:
                        if args.reg == 'l1':
                            freq_reg += torch.abs(param).sum()
                        else:
                            freq_reg += torch.square(param).sum().sqrt()

                loss += alpha * freq_reg

        # backpropagate loss
        try:
            loss.backward()
        except RuntimeError:
            pass
        optimizer.step()
        scheduler.step()
        renderer.step(model)
        optimizer.zero_grad()

        # log metrics
        if not args.debug and k % args.val_rate != 0:
            wandb.log({
                'train_psnr': psnr,
                'lr': scheduler.lr,
                'alpha': alpha
            })

        # toggle ensemble
        if args.model == 'ensemble':
            if k in milestones:
                model.toggle_ensemble(m)
                m += 1

        # compute validation
        compute_val = k % args.val_rate == 0 and k > 0 and args.val
        if compute_val:
            model.eval()
            renderer.eval()
            #lpips_net.eval()
            with torch.no_grad():
                val_metrics = validation(
                        hwf,
                        model,
                        renderer,
                        lpips_net,
                        val_loader,
                        device
                )
                val_psnr, val_ssim, val_lpips = val_metrics

                # render test image
                rgb, depth = renderer.render_poses(hwf, testpose, 
                                                   model, ndc, device)

                # log data to wandb
                if not args.debug:
                    wandb.log({
                        'train_psnr': psnr,
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

def main():
    # select device
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # print device info or abort if no CUDA device available
    print(f"Device: {torch.cuda.get_device_name(device)}")

    if not args.debug:
        wandb.login()
        # set up wandb run to track training
        name = f"{args.model}"
        name = name + f"-{args.reg}" if args.ao is not None else name
        name = name + f"-ao={args.ao:.2e}" if args.ao is not None else name
        run = wandb.init(
            project='fs-nerf',
            name=name,
            config=args
        )

    # training/validation datasets
    dataset_dict = {
            'synthetic': 
                (D.SyntheticRealistic, 
                {'white_bkgd': args.white_bkgd}),
            'llff': 
                (D.LLFF, 
                {'factor': args.factor, 
                 'bd_factor': args.bd_factor, 
                 'recenter': not args.no_recenter, 
                 'ndc': True}),
    }
    dataset_name, dataset_kwargs = dataset_dict[args.dataset]
    train_set = dataset_name(
            args.scene,
            'train',
            n_imgs=args.n_imgs,
            img_mode=False,
            **dataset_kwargs
    )
    nval_imgs = 25 if args.dataset == 'synthetic' else 8
    subset_size = int(args.val_ratio * nval_imgs) # % of val samples
    val_set = dataset_name(
            args.scene,
            'val',
            n_imgs=subset_size,
            img_mode=True,
            **dataset_kwargs
    )
    # data loader(s)
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
    )
    val_loader = DataLoader(
            val_set,
            batch_size=len(val_set),
            shuffle=True,
            num_workers=8
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
    t = 1 if args.dataset == 'llff' else 5
    factor = 1 if args.dataset == 'llff' else 0
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-t, t]),
            yaxis=dict(range=[-t, t]),
            zaxis=dict(range=[-t*factor, t]),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    if not args.debug:
        wandb.log({
            'camera_positions': fig,
            'rgb_gt': wandb.Image(
                train_set.testimg.numpy(),
                caption='Ground Truth RGB'
            )
        })

    if not args.render_only:
        # initialize modules
        model, renderer, lpips_net = init_models(train_set)
        model.to(device)
        renderer.to(device)
        #lpips_net.to(device)

        # train model
        train(
                model, 
                renderer,
                lpips_net,
                train_loader,
                val_loader,
                device=device
        )
        # final validation set and loader
        val_set = dataset_name(
                args.scene,
                'val',
                n_imgs=nval_imgs,
                img_mode=True,
                **dataset_kwargs
        )
        val_loader = DataLoader(
                val_set,
                batch_size=len(val_set),
                shuffle=True,
                num_workers=8
        )
        # compute final validation metrics
        model.eval()
        renderer.eval()
        lpips_net.eval()
        with torch.no_grad():
            val_metrics = validation(
                    train_set.hwf,
                    model,
                    renderer,
                    lpips_net,
                    val_loader,
                    device
            )
        # log final metrics
        final_psnr, final_ssim, final_lpips = val_metrics
        
        if not args.debug:
            wandb.log({
                'final_psnr': final_psnr,
                'final_ssim': final_ssim,
                'final_lpips': final_lpips
            })
    else:
        model = init_models()
        # load model
        model.load_state_dict(torch.load(out_dir + '/model/nn.pt'))

    if not args.debug:
        # build base path for output directories
        out_dir = os.path.normpath(
                os.path.join(
                    args.out_dir, 
                    args.model, 
                    args.dataset,
                    args.scene,
                    f"n_imgs_{str(args.n_imgs)}",
                    run.id
                )
        )

        # create output directories
        folders = ['video', 'model']
        [os.makedirs(os.path.join(out_dir, f), exist_ok=True) for f in folders]
        # save model
        if not args.render_only:
            torch.save(model.state_dict(), out_dir + '/model/nn.pt')

    # compute path poses for video output
    path_poses = train_set.path_poses

    # render frames for poses
    model.eval()
    renderer.eval()
    
    # render maps
    renderer.set_chunksize(2*args.batch_size)
    frames, d_frames = renderer.render_poses(train_set.hwf,
                                             path_poses,
                                             model,
                                             train_set.ndc,
                                             device)

    if not args.debug:
        # put together frames and save result into .mp4 file
        R.render_video(
                basedir=f'{out_dir}/video/',
                frames=frames,
                d_frames=d_frames
        )
        # log final video renderings to wandb
        wandb.log({
            'rgb_video': wandb.Video(f'{out_dir}/video/rgb.mp4', fps=30),
            'depth_video': wandb.Video(f'{out_dir}/video/depth.mp4', fps=30)
        })

if __name__ == '__main__':
    main()
