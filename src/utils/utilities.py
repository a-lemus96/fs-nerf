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

    # use pinhole model to represent grid in terms of camera coordinate frame
    focal = focal_length.item()
    directions = torch.stack([(i - width * 0.5) / focal, 
                               -(j - height * 0.5) / focal,
                               -torch.ones_like(i)], dim=-1)

    # normalize directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    if camera_pose is None:
        return directions

    # apply camera rotation to ray directions
    directions_world = torch.sum(directions[..., None, :] * camera_pose[:3, :3], axis=-1)
    # apply camera translation to ray origin
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
