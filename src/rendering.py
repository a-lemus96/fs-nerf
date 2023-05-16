# stdlib modules
import argparse
import os
from typing import Callable, Optional, Tuple

# third-party modules
import imageio
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch 
from torch import nn
from tqdm import tqdm

# custom modules
from utilities import *

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

def sphere_path(radius: float = 3.,
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

def render_path(
    render_poses: torch.Tensor,
    near: float,
    far: float,
    hwf: torch.Tensor,
    encode: Callable[[torch.Tensor], torch.Tensor], 
    model: nn.Module,
    kwargs_sample_stratified: dict = None,
    n_samples_hierarchical: int = 0,
    kwargs_sample_hierarchical: dict = None,
    fine_model: nn.Module = None,
    encode_viewdirs: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    chunksize: int = 2**12
    ) -> Tuple[torch.Tensor]:
    r"""Render video outputs for the incoming camera poses.
    ----------------------------------------------------------------------------
    Args:
        render_poses: [N, 4, 4]. Camera poses to render from.
        chunksize: int. Size of smaller minibatches to avoid OOM.
        hwf: [3]. Height, width and focal length.
    Returns:
        frames: [N, H, W, 3]. Rendered RGB frames.
        d_frames: (N, H, W)-shape tensor. Rendered depth frames."""

    H, W, focal = hwf
    model.eval()

    frames, d_frames = [], []
    print("Rendering frames...")
    for i, pose in enumerate(tqdm(render_poses)):
        with torch.no_grad():
            # Get rays
            rays_o, rays_d = get_rays(H, W, focal, pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])

            # Compute NeRF forward pass
            outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=n_samples_hierarchical,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=chunksize)

            # read predicted rgb frame
            rgb_map = outputs['rgb_map']
            rgb_map = rgb_map.reshape([H, W, 3]).detach().cpu().numpy()
            
            # read predicted depth frame
            depth_map = outputs['depth_map']
            depth_map = depth_map.reshape([H, W]).detach().cpu().numpy()

        # append rgb and depth frames
        frames.append(rgb_map)
        d_frames.append(depth_map)

    # stack all frames in numpy arrays
    frames = np.stack(frames, 0)
    d_frames = np.stack(d_frames, 0)

    return frames, d_frames

def render_video(
    basedir: str,
    frames: torch.Tensor,
    d_frames: torch.Tensor
    ) -> None:
    r"""Video rendering functionality. It takes a series of frames and joins
    them in .mp4 files.
    ----------------------------------------------------------------------------
    Args:
        basedir: str. Base directory where to store .mp4 file
        frames: [N, H, W, 3]. N rgb frames
        d_frames: (N, H, W)-shape. N depth frames
    Returns:
        None"""
    # rgb video output
    imageio.mimwrite(basedir + 'rgb.mp4', to8b(frames), fps=30, quality=8)
    
    # map depth values to RGBA using a colormap
    norm = matplotlib.colors.Normalize(vmin=np.amin(d_frames),
                                       vmax=np.amax(d_frames))
    mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
    # flatten d_frames before applying mapping
    d_rgba = mapper.to_rgba(d_frames.flatten())
    # return to normal dimensions
    d_rgba = np.reshape(d_rgba, list(d_frames.shape[:3]) + [-1])
    print(d_rgba.shape)
    # unflatten d_frames
    imageio.mimwrite(basedir + 'd.mp4', to8b(d_rgba), fps=30, quality=8)
