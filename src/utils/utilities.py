# stdlib modules
import os
from typing import Optional, Tuple, List, Union, Callable

# third-party modules
from nerfacc.volrend import rendering
from nerfacc.estimators.occ_grid import OccGridEstimator
import numpy as np
import torch
from torch import nn
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
        pose: Tensor,
        hwf: Tuple[int, int, float],
        device: torch.device = torch.device('cpu'),
) -> Tuple[Tensor, Tensor]:
    """
    Computes ray origins and directions in world coordinates for a given camera 
    pose.
    ----------------------------------------------------------------------------
    Args:
        pose: [4, 4]. Camera pose matrix.
        hwf: [3]. Height, width, focal length.
        device: Device to use for computation.
    Returns:
        origins_w: [..., 3]. Ray origins in world coords.
        dirs_w: [..., 3]. Ray directions in world coords.
    ----------------------------------------------------------------------------
    """
    H, W, focal = hwf # unpack intrinsics
    pose = pose.to(device)
    # create grid of coordinates
    i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing='ij'
    )
    i, j = torch.transpose(i, -1, -2), torch.transpose(j, -1, -2)
    # use pinhole model to map grid into camera space
    f = focal
    dirs = torch.stack(
            [(i - W*0.5)/f, -(j - H*0.5)/f, -torch.ones_like(i)], 
            dim=-1
    )
    # normalize directions
    dirs = dirs/torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs[None, ..., None, :]
    # apply camera rotation to ray directions
    poses = poses.unsqueeze(-3) if poses.dim() == 2 else poses
    poses = poses[..., None, None, :3, :3]
    dirs_w = torch.sum(dirs * poses, axis=-1)
    # apply camera translation to ray origin
    origins_w = poses[..., :3, -1].expand_as(dirs_w)

    return origins_w, dirs_w 

def to_ndc(
        rays_o: Tensor,
        rays_d: Tensor,
        hwf: Tuple[int, int, float],
        near: float
    ) -> Tuple[Tensor, Tensor]:
    """
    Convert rays from world coordinates to normalized device coordinates.
    ----------------------------------------------------------------------------
    Args:
        rays_o: [..., 3]. Ray origins in world coordinates.
        rays_d: [..., 3]. Ray directions in world coordinates.
        near: near plane distance.
        hwf: [3]. Height, width, focal length.
    Returns:
        ndc_o: [..., 3]. Ray origins in normalized device coords.
        ndc_d: [..., 3]. Ray directions in normalized device coords.
    """
    H, W, focal = hwf   # unpack intrinsics
    # shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # perform projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    # stack origins and directions
    ndc_o = torch.stack([o0, o1, o2], dim=-1)
    ndc_d = torch.stack([d0, d1, d2], dim=-1)

    return ndc_o, ndc_d

def get_chunks(inputs: Tensor, chunksize: int) -> List[Tensor]:
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
