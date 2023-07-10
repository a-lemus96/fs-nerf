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

def sphere_path(radius: float = 3.5,
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
    hwf: torch.Tensor,
    chunksize: int,
    model: nn.Module,
    pos_fn: Callable[[torch.Tensor], torch.Tensor], 
    dir_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    white_bkgd: bool = False
) -> Tuple[torch.Tensor]:
    """Renders a video from a given path of camera poses.
    ----------------------------------------------------------------------------
    Args:
        render_poses: (frames, 4, 4)-shape tensor containing poses to render
        hwf: [3]-shape tensor containing height, width and focal length
        chunksize int: Number of rays to render in parallel
        model: nn.Module. NeRF model to use for rendering
        pos_fn: Callable. Pos encoding for spatial inputs
        dir_fn: Optional Callable. Pos encoding for direction inputs
        white_bkgd: bool. Whether to use white background
    Returns:
        frames: [N, H, W, 3]. N rgb frames
    ----------------------------------------------------------------------------
    """
    H, W, focal = hwf

    frames, d_frames = [], []
    print("Rendering frames...")
    for i, pose in enumerate(tqdm(render_poses)):
        with torch.no_grad():
            # render frame
            rgb, depth = U.render_frame(
                    H, W, focal, pose,
                    args.batch_size,
                    pos_fn, model,
                    dir_fn=dir_fn,
                    white_bkgd=args.white_bkgd,
                    estimator=estimator,
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
    d_frames: torch.Tensor
) -> None:
    """Video rendering functionality. It takes a series of frames and joins
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
    # unflatten d_frames
    imageio.mimwrite(basedir + 'd.mp4', to8b(d_rgba), fps=30, quality=8)
