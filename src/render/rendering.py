# stdlib modules import argparse
import os
from typing import Callable, Optional, Tuple

# third-party modules
import imageio
from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch 
from torch import nn
from torch import Tensor
from tqdm import tqdm

# custom modules
import utils.utilities as U


# Function to map float values to [0, 255] integer range
to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)


def render_rays(
        rays_o: Tensor,
        rays_d: Tensor,
        estimator: OccGridEstimator,
        model: nn.Module,
        train: bool = False,
        white_bkgd: bool = False,
        render_step_size: float = 5e-3,
        device: torch.device = torch.device('cpu')
) -> Tuple[Tensor]:
    """Renders rays using a given NeRF model.
    ----------------------------------------------------------------------------
    Args:
        rays_o: (n_rays, 3)-shape tensor containing ray origins
        rays_d: (n_rays, 3)-shape tensor containing ray directions
        estimator: OccGridEstimator object
        model: NeRF model
        train: Whether to train model
        white_bkgd: Whether to use white background
        render_step_size: Rendering step size
        device: Device to use for rendering
    Returns:
        rgb: (n_rays, 3)-shape tensor containing RGB values
        opacity: (n_rays,)-shape tensor containing opacity values
        depth: (n_rays,)-shape tensor containing depth values
        extras: (n_rays,)-shape tensor containing extra values
        ray_indices: (n_rays,)-shape tensor containing ray indices
    ----------------------------------------------------------------------------
    """
    # send rays to device
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)

    def sigma_fn(t_starts, t_ends, ray_indices):
        to = rays_o[ray_indices]
        td = rays_d[ray_indices]
        x = to + td * (t_starts + t_ends)[:, None] / 2.0
        sigmas = model(x)

        return sigmas.squeeze(-1)

    ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            render_step_size=render_step_size,
            stratified=train,
            near_plane=0.,
            far_plane=1e10
    )

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            to = rays_o[ray_indices]
            td = rays_d[ray_indices]
            x = to + td * (t_starts + t_ends)[:, None] / 2.0
            out = model(x, td)
            rgbs = out[..., :3]
            sigmas = out[..., -1]

            return rgbs, sigmas.squeeze(-1)

    render_bkgd = white_bkgd * torch.ones(
            (3,), 
            device=device, 
            requires_grad=train
    )

    try:
        output = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=len(rays_o),
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd
        )
    except AssertionError as e:
        output = (
                torch.ones_like(rays_o) * white_bkgd,
                None,
                torch.zeros_like(rays_o[:, 0].unsqueeze(1), dtype=torch.float32),
                None,
        )

    return output, ray_indices


def render_frame(
        hwf: Tuple[int, int, float],
        near: float,
        far: float,
        pose: torch.Tensor, 
        chunksize: int,
        estimator: OccGridEstimator,
        model: nn.Module,
        train: bool = False,
        ndc: bool = False,
        white_bkgd: bool = False,
        render_step_size: float = 5e-3,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Render an image from a given pose. Camera rays are chunkified to avoid memo-
    ry issues.
    ----------------------------------------------------------------------------
    Args:
        hwf: (3,)-shape tuple containing height, width and focal length
        near: Near bound
        far: Far bound
        pose: Camera pose
        chunksize: Chunk size for rays
        estimator: OccGridEstimator object
        model: NeRF model
        ndc: Whether to use normalized device coordinates
        train: Whether to train model
        white_bkgd: Whether to use white background
        render_step_size: Rendering step size
        device: Device to use for rendering
    Returns:
        img: (H, W, 3)-shape tensor containing RGB values
        depth_map: (H, W)-shape tensor containing depth values
    """
    H, W, _ = hwf
    rays_o, rays_d = U.get_rays(pose, hwf, device) # compute rays
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3) # flatten rays
    if ndc:
        # convert rays to normalized device coordinates
        rays_o, rays_d = U.to_ndc(rays_o, rays_d, hwf, 1.)

    # chunkify rays to avoid memory issues
    chunked_rays_o = U.get_chunks(rays_o, chunksize=chunksize)
    chunked_rays_d = U.get_chunks(rays_d, chunksize=chunksize)

    # compute image and depth in chunks
    img = []
    depth_map = []
    for rays_o, rays_d in zip(chunked_rays_o, chunked_rays_d):
        out = render_rays(
                rays_o, 
                rays_d,
                estimator,
                model,
                white_bkgd,
                render_step_size=render_step_size,
                device=device,
        )
        (rgb, _, depth, _), _ = out 
        img.append(rgb)
        depth_map.append(depth)

    # aggregate chunks
    img = torch.cat(img, dim=0)
    depth = torch.cat(depth_map, dim=0).clamp(near, far)

    return img.reshape(H, W, 3), depth.reshape(H, W)
        

def render_path(
        render_poses: torch.Tensor,
        hwf: Tuple[int, int, float],
        near: float,
        far: float,
        chunksize: int,
        model: nn.Module,
        estimator: OccGridEstimator,
        ndc: bool = False,
        train: bool = False,
        white_bkgd: bool = False,
        render_step_size: float = 5e-3,
        device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor]:
    """Renders a video from a given path of camera poses.
    ----------------------------------------------------------------------------
    Args:
        render_poses: (frames, 4, 4)-shape tensor containing poses to render
        hwf: (3,)-shape tuple containing height, width and focal length
        near: Near bound
        far: Far bound
        chunksize int: Number of rays to render in parallel
        model: nn.Module. NeRF model to use for rendering
        estimator: OccGridEstimator. OccGridEstimator object
        ndc: bool. Whether to use normalized device coordinates
        train: bool. Whether to train model
        white_bkgd: bool. Whether to use white background
        render_step_size: float. Step size for rendering
        device: torch.device. Device to use for rendering
    Returns:
        frames: [N, H, W, 3]. N rgb frames
    ----------------------------------------------------------------------------
    """
    H, W, _ = hwf
    frames, d_frames = [], []
    pbar = tqdm(render_poses, desc=f"[Rendering Frames]")
    for i, pose in enumerate(pbar):
        with torch.no_grad():
            # render frame
            rgb, depth = render_frame(
                    hwf,
                    near, far, pose,
                    chunksize,
                    estimator,
                    model,
                    train=train,
                    ndc=ndc,
                    white_bkgd=white_bkgd,
                    render_step_size=render_step_size,
                    device=device
            )

            # read predicted rgb frame
            rgb = rgb.reshape([H, W, 3]).detach().cpu().numpy()
            
            # read predicted depth frame
            depth = depth.reshape([H, W]).detach().cpu().numpy()

        # append rgb and depth frames
        frames.append(rgb)
        d_frames.append(depth)

    # stack all frames in numpy arrays
    frames = np.stack(frames, 0)
    d_frames = np.stack(d_frames, 0)

    return frames, d_frames

def render_video(
    basedir: str,
    frames: torch.Tensor,
    d_frames: torch.Tensor,
    cmap: str = 'plasma'
) -> None:
    """Video rendering functionality. It takes a series of frames and joins
    them in .mp4 files.
    ----------------------------------------------------------------------------
    Args:
        basedir: str. Base directory where to store .mp4 file
        frames: [N, H, W, 3]. N rgb frames
        d_frames: (N, H, W)-shape. N depth frames
        cmap: str. Colormap to use for depth frames
    Returns:
        None"""
    # rgb video output
    imageio.mimwrite(basedir + 'rgb.mp4', to8b(frames), fps=30, quality=8)
    
    # map depth values to RGBA using a colormap
    norm = matplotlib.colors.Normalize(vmin=np.amin(d_frames),
                                       vmax=np.amax(d_frames))
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    # flatten d_frames before applying mapping
    d_rgba = mapper.to_rgba(d_frames.flatten())
    # return to normal dimensions
    d_rgba = np.reshape(d_rgba, list(d_frames.shape[:3]) + [-1])
    # unflatten d_frames
    imageio.mimwrite(basedir + 'depth.mp4', to8b(d_rgba), fps=30, quality=8)
