# stdlib modules
import os
from typing import Optional, Tuple, List, Union, Callable

# third-party modules
from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator
import numpy as np
import torch
from torch import nn
from torch import jit
from torch import Tensor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def save_origins_and_dirs(poses):
    '''Plot and save optical axis positions and orientations for each camera pose.
    Args:
        poses: [num_poses, 4, 4]. Camera poses.
    Returns:
        None
    '''
    # Compute optical axis orientations and positions
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses]) 
    origins = poses[:, :3, -1]
    # Plot 3D arrows representing position and orientation of the cameras
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(origins[..., 0].flatten(), origins[..., 1].flatten(), origins[..., 2].flatten(),
                  dirs[..., 0].flatten(), dirs[..., 1].flatten(), dirs[..., 2].flatten(),
                  length=0.5,
                  normalize=True)
    plt.savefig('out/verify/poses.png')
    plt.close()

# RAY HELPERS

def get_rays(
        height: int,
        width: int,
        focal_length: float,
        camera_pose: torch.Tensor=None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Find origin and direction of rays through every pixel and camera origin.
    Args:
        height: Image height.
        width: Image width.
        focal_length: Focal length of the camera.
        camera_pose: [4, 4]. Camera pose matrix.
    Returns:
        origins_world: [height, width, 3]. Coordinates of rays using world coordinates.
        directions_world: [height, width, 3]. Orientations of rays in world coordinates.
    '''
    # create grid of coordinates
    i, j = torch.meshgrid(torch.arange(width, dtype=torch.float32).to(camera_pose), 
                          torch.arange(height, dtype=torch.float32).to(camera_pose), 
                          indexing='ij')
    i, j = torch.transpose(i, -1, -2), torch.transpose(j, -1, -2)

    # Use pinhole camera model to represent grid in terms of camera coordinate frame
    focal = focal_length.item()
    directions = torch.stack([(i - width * 0.5) / focal, 
                               -(j - height * 0.5) / focal,
                               -torch.ones_like(i)], dim=-1)

    # Normalize directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    if camera_pose is None:
        return directions

    # Apply camera rotation to ray directions
    directions_world = torch.sum(directions[..., None, :] * camera_pose[:3, :3], axis=-1) 
    # Apply camera translation to ray origin
    origins_world = camera_pose[:3, -1].expand(directions_world.shape)

    return origins_world, directions_world 

def get_chunks(inputs: torch.Tensor,
               chunksize: int) -> List[torch.Tensor]:
    """
    Split inputs into chunks of size chunksize.
    ----------------------------------------------------------------------------
    Args:
        inputs: tensor to be chunkified
        chunksize: size of each chunk
    Returns:
        list of tensors of size chunksize
    """
    inds = range(0, inputs.shape[0], chunksize)

    return [inputs[i:i + chunksize] for i in inds]

def render_frame(
        H: int,
        W: int,
        focal: float,
        pose: torch.Tensor, 
        chunksize: int,
        pos_fn: Callable[[torch.Tensor], torch.Tensor],
        model: nn.Module,
        dir_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        white_bkgd = False,
        estimator = OccGridEstimator,
        render_step_size = None,
        device = None
        ) -> torch.Tensor:
    """Render an image from a given pose. Camera rays are chunkified to avoid
    memory issues.
    ----------------------------------------------------------------------------
    Args:
        H: height of the image
        W: width of the image
        focal: focal length of the camera
        pose: [4, 4] tensor with the camera pose
        chunksize: size of the chunks to be processed
        pos_fn: positional encoding for spatial inputs
        model: coarse model
        dir_fn: positional encoding for directional inputs
        white_bkgd: whether to use a white background
    Returns:
        img: [H, W, 3] tensor with the rendered image
        depth: [H, W] tensor with the depth map
    """
    rays_o, rays_d = get_rays(H, W, focal, pose) # compute rays
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3) # flatten rays

    # chunkify rays to avoid memory issues
    chunked_rays_o = get_chunks(rays_o, chunksize=chunksize)
    chunked_rays_d = get_chunks(rays_d, chunksize=chunksize)

    # compute image and depth in chunks
    img = []
    depth_map = []
    for rays_o, rays_d in zip(chunked_rays_o, chunked_rays_d):
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
                stratified=False
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

        rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=torch.tensor(
                    white_bkgd * torch.ones(3),
                    device=device,
                    requires_grad=True
                )
        )
        img.append(rgb)
        depth_map.append(depth)

    # aggregate chunks
    img = torch.cat(img, dim=0)
    depth = torch.cat(depth_map, dim=0)

    return img.reshape(H, W, 3), depth.reshape(H, W)
