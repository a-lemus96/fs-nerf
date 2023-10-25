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

# FUNCTIONS TO COMPUTE POSES FROM SPHERICAL COORDINATES

# Translate across world's z-axis
trans_t = lambda t: torch.Tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., t],
                                 [0., 0., 0., 1.]]).float() 

# Rotate around world's x-axis
rot_theta = lambda theta: torch.Tensor([[1., 0., 0., 0.],
                                        [0., np.cos(theta), -np.sin(theta), 0.],
                                        [0., np.sin(theta), np.cos(theta), 0.],
                                        [0., 0., 0., 1.]]).float()

# Rotate around world's z-axis
rot_phi = lambda phi: torch.Tensor([[np.cos(phi), -np.sin(phi), 0., 0.],
                                    [np.sin(phi), np.cos(phi), 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]]).float()

def pose_from_spherical(
    radius: float, 
    theta: float,
    phi: float
    ) -> torch.Tensor:
    r"""Computes 4x4 camera pose from 3D location expressed in spherical coords.
    Camera frame points toward object with its y-axis tangent to the virtual
    spherical surface defined by the given radius.
    ---------------------------------------------------------------------------- 
    Args:
        radius: float. Sphere radius.
        theta: 0° < float < 90°. colatitude angle.
        phi: 0° < float < 360°. Azimutal angle.
    Returns:
        pose: [4, 4]. Camera to world transformation."""

    pose = trans_t(radius) 
    pose = rot_theta(theta/180. * np.pi) @ pose
    pose = rot_phi(phi/180. * np.pi) @ pose 
    
    return pose

# RENDERING PATH UTILITIES

def sphere_path(
        radius: float = 4.0311289,
        theta: float = 45.,
        frames: int = 40
) -> torch.Tensor:
    r"""Computes set of frames for inward facing camera poses using constant
    radius and theta, while varying azimutal angle within the interval [0,360].
    ----------------------------------------------------------------------------
    Args:
        radius: Sphere radius
        theta: 0° < float < 90°. Colatitude angle
        frames: Number of samples along azimutal interval
    Returns:
        render_poses: (frames, 4, 4)-shape tensor containing poses to render"""

    # Compute camera poses along video rendering path
    render_poses = [pose_from_spherical(radius, theta, phi)
                    for phi in np.linspace(0, 360, frames, endpoint=False)]
    render_poses = torch.stack(render_poses, 0)
    return render_poses

# VIDEO RENDERING UTILITIES

# Function to map float values to [0, 255] integer range
to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

def render_rays(
        rays_o: Tensor,
        rays_d: Tensor,
        estimator: OccGridEstimator,
        device: str,
        model: nn.Module,
        train: bool = False,
        white_bkgd: bool = False,
        render_step_size: float = 5e-3
) -> Tuple[Tensor]:
    """Renders rays using a given NeRF model.
    ----------------------------------------------------------------------------
    Args:
        rays_o: (n_rays, 3)-shape tensor containing ray origins
        rays_d: (n_rays, 3)-shape tensor containing ray directions
        estimator: OccGridEstimator object
        device: Device to use for rendering
        model: NeRF model
        train: Whether to train model
    Returns:
        rgb: (n_rays, 3)-shape tensor containing RGB values
        opacity: (n_rays,)-shape tensor containing opacity values
        depth: (n_rays,)-shape tensor containing depth values
        extras
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
            far_plane=8.
    )

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            to = rays_o[ray_indices]
            td = rays_d[ray_indices]
            x = to + td * (t_starts + t_ends)[:, None] / 2.0
            out = model(x, td)
            rgbs = out[..., :3]
            sigmas = out[..., -1]

            return rgbs, sigmas.squeeze(-1)

    render_bkgd = white_bkgd * torch.ones((3,), device=device, requires_grad=train)

    try:
        output = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
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

    return output


def render_frame(
        H: int,
        W: int,
        focal: float,
        pose: torch.Tensor, 
        chunksize: int,
        estimator: OccGridEstimator,
        device: str,
        model: nn.Module,
        train: bool = False,
        white_bkgd: bool = False,
        render_step_size: float = 5e-3
        ) -> torch.Tensor:
    """Render an image from a given pose. Camera rays are chunkified to avoid
    memory issues.
    ----------------------------------------------------------------------------
    ----------------------------------------------------------------------------
    """
    rays_o, rays_d = U.get_rays(H, W, focal, pose) # compute rays
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3) # flatten rays

    # chunkify rays to avoid memory issues
    chunked_rays_o = U.get_chunks(rays_o, chunksize=chunksize)
    chunked_rays_d = U.get_chunks(rays_d, chunksize=chunksize)

    # compute image and depth in chunks
    img = []
    depth_map = []
    for rays_o, rays_d in zip(chunked_rays_o, chunked_rays_d):
        rgb, opacity, depth, extras = render_rays(
                rays_o, rays_d,
                estimator,
                device,
                model,
                white_bkgd,
                render_step_size=render_step_size
        )

        img.append(rgb)
        depth_map.append(depth)

    # aggregate chunks
    img = torch.cat(img, dim=0)
    depth = torch.cat(depth_map, dim=0)

    return img.reshape(H, W, 3), depth.reshape(H, W)
        

def render_path(
        render_poses: torch.Tensor,
        hwf: torch.Tensor,
        chunksize: int,
        device: str,
        model: nn.Module,
        estimator: OccGridEstimator,
        train: bool = False,
        white_bkgd: bool = False,
        render_step_size: float = 5e-3
) -> Tuple[torch.Tensor]:
    """Renders a video from a given path of camera poses.
    ----------------------------------------------------------------------------
    Args:
        render_poses: (frames, 4, 4)-shape tensor containing poses to render
        hwf: [3]-shape tensor containing height, width and focal length
        chunksize int: Number of rays to render in parallel
        device: str. Device to use for rendering
        model: nn.Module. NeRF model to use for rendering
        train: bool. Whether to train model
        white_bkgd: bool. Whether to use white background
        render_step_size: float. Step size for rendering
    Returns:
        frames: [N, H, W, 3]. N rgb frames
    ----------------------------------------------------------------------------
    """
    H, W, focal = hwf

    frames, d_frames = [], []
    pbar = tqdm(render_poses, desc=f"[Rendering Frames]")
    for i, pose in enumerate(pbar):
        with torch.no_grad():
            # render frame
            rgb, depth = render_frame(
                    H, W, focal, pose,
                    chunksize,
                    estimator,
                    device,
                    model,
                    train,
                    white_bkgd=white_bkgd,
                    render_step_size=render_step_size,
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
