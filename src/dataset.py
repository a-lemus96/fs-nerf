# standard library modules
import os
from typing import Tuple, List, Union, Callable

# third-party modules
import torch
from torch.utils.data import Dataset

# custom modules
from dataload import *
from utilities import *

# NERF DATASET

class NerfDataset(Dataset):
    r"""NeRF dataset. A NeRF dataset consists of N x H x W ray origins and
    directions relative to the world frame. Here, N is the number of training
    images of size H x W."""
    def __init__(
        self,
        dataset: str,
        subset: str,
        scene: str,
        n_imgs: int,
        test_idx: int,
        f_forward: bool = False,
        factor: int = None,
        near: int = 2.,
        far: int = 7.):

        # Initialize attributes
        self.n_imgs = n_imgs
        self.test_idx = test_idx
        self.near = near # near and far sampling bounds for each ray
        self.far = far

        # Load images and camera poses
        reader = DataReader(dataset, subset, scene, factor)
        imgs, poses, hwf, _, _ = reader.get_data()

        # Validation image
        self.testimg = imgs[test_idx]
        self.testpose = poses[test_idx]

        # Selection indices
        self.inds = torch.arange(n_imgs)

        if f_forward:
            # Set central view arbitrarily
            central = 15
            # Retrieve camera origins
            origins = poses[:, :3, 3]
            # Compute camera distances to central camera
            dists = torch.sum((origins[central] - origins)**2, -1)
            # Sort distances and retrieve indices
            _, select = torch.sort(dists)

            # Select validation image and pose
            self.test_idx = select[n_imgs]
            self.testimg = imgs[self.test_idx]
            self.testpose = poses[self.test_idx]

            # Select training images and poses
            self.inds = select[:n_imgs]
            imgs = imgs[self.inds]
            poses = poses[self.inds]

        H, W, self.focal = hwf
        self.H, self.W = int(H), int(W)
        N = self.inds.shape[0]
        
        # Get rays
        self.rays = torch.stack([torch.stack(
                                 get_rays(self.H, self.W, self.focal, p), 0)
                                 for p in poses[:N]], 0)

        # Append RGB supervision and local rays dirs info
        rays_rgb = torch.cat([self.rays,
                              imgs[:N, None]], 1) 

        # Rearrange data and reshape
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, rays_rgb.shape[3], 3])
        rays_rgb = torch.transpose(rays_rgb, 0, 1)
        
        self.rays_rgb = rays_rgb.type(torch.float32)

    def __len__(self):
        return self.rays_rgb.shape[1]

    def __getitem__(self, idx):
        rays_o, rays_d, target_pixs = self.rays_rgb[:, idx]
        
        return rays_o, rays_d, target_pixs

# DEPTH SUPERVISED NERF DATASET

class DSNerfDataset(NerfDataset):
    r"""Depth supervised NeRF dataset. It consists of N x H x W ray origins and
    directions relative to the world frame. Here, N is the number of training
    images of size H x W. It also contains depth information associated to each
    ray."""
    def __init__(
        self,
        dataset: str,
        subset: str,
        scene: str,
        n_imgs: int,
        test_idx: int,
        f_forward: bool = False,
        factor: int = None,
        near: int=2.,
        far: int=7.): 
        # Call base class constructor method
        super().__init__(dataset, subset, scene, n_imgs, test_idx,
                         f_forward, factor, near, far)

        # Load images, camera poses and depth maps
        reader = DataReader(dataset, subset, scene, factor)
        imgs, poses, hwf, maps, backs = reader.get_data()

        test_back = backs[self.test_idx].type(torch.bool)
        test_map = maps[self.test_idx]
        backs = backs[self.inds]
        maps = maps[self.inds]
        N = backs.shape[0]
 
        # Local rays
        self.local_dirs = get_rays(self.H, self.W, self.focal, local_only=True)

        # Compute depth along rays for test depth map
        #test_map = -test_map / self.local_dirs[..., -1]
        self.test_map = test_map.type(torch.float32) * (~test_back)

        # Expanded version of local rays according to number of training imgs
        local_dirs = self.local_dirs[None, None, ...].expand(N, 1, self.H,
                                                             self.W, 3)

        # Compute depth along rays
        t_maps = -maps[:, None, ...] / local_dirs[..., -1]

        # Compute background masks and concatenate with depth maps
        backs = backs[:, None, ..., None]
        depths = torch.cat((t_maps[..., None], backs), 1)

        depths = torch.permute(depths, [0, 2, 3, 1, 4])
        depths = depths.reshape([-1, depths.shape[3], 1]).type(torch.float32)
        depths = torch.transpose(depths, 0, 1)
        self.depths = depths.type(torch.float32)

    def __getitem__(self, idx):
        rays_o, rays_d, target_pixs = self.rays_rgb[:, idx]
        depths, backs = self.depths[:, idx, 0]

        return rays_o, rays_d, target_pixs, depths, backs
