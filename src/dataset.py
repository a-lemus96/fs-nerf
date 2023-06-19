# standard library modules
import json
import os
from typing import Tuple, List, Union, Callable

# third-party modules
import imageio as iio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Resize

# custom modules
import utilities as utils

class SyntheticRealistic(Dataset):
    """
    Synthetic realistic dataset. It is made up of N x H x W ray origins and
    directions in world coordinate frame paired with ground truth pixel colors.
    Here, N is the number of training images of size H x W.
    ----------------------------------------------------------------------------
    """
    def __init__(self, scene: str, root: str, factor: float=None) -> None:
        """
        Initialize the dataset.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene name
            root (str): root directory. It can be either 'train', 'val', or 
                        'test'
            factor (float): factor to scale images and camera intrinsics
        Returns:
            None
        """
        super(SyntheticRealistic).__init__()  # inherit from Dataset
        self.scene = scene
        self.root = root
        self.factor = factor

        # load the dataset
        imgs, poses, hwf = self.__load()

        # compute rays
        H, W, f = hwf
        rays = torch.stack([torch.cat(utils.get_rays(H, W, f, p), -1) 
                            for p in poses], 0)
        rays = rays.reshape(-1, 6)
        self.rays_o = rays[:, :3]
        self.rays_d = rays[:, 3:]

        # add pixel colors
        self.rgba = imgs.reshape(-1, 4)

    def __len__(self) -> int:
        """Compute the number of training samples.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            N (int): number of training samples
        """
        return self.rgb.shape[0]
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get a training sample by index.
        ------------------------------------------------------------------------
        Args:
            idx (int): index of the training sample
        Returns:
            ray_o (Tensor): [3,]. Ray origin
            ray_d (Tensor): [3,]. Ray direction
            rgba (Tensor): [4,]. Pixel RGBa color
        """
        return self.rays_o[idx], self.rays_d[idx], self.rgba[idx]


    def __factor(self, imgs: Tensor, hwf: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Downsample images and apply resize factor to camera intrinsics.
        ------------------------------------------------------------------------
        Args:
            imgs: [N, H, W, 4]. RGBa images
            hwf: [3,]. Camera intrinsics.
        Returns:
            new_imgs: [N, H, W, 4]. Downsampled images
            new_hwf: [3,]. Updated camera intrinsics
        """
        # apply factor to camera intrinsics
        H, W, f = hwf
        new_W, new_H = H // factor, W // factor
        new_focal = hwf[2] / float(factor)
        new_hwf = torch.Tensor((new_H, new_W, new_focal))

        # downsample images
        new_imgs = Resize((new_H, new_W))(imgs)

        return new_imgs, new_hwf

    def __load(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Load the dataset.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            imgs: [N, H, W, 4]. RGBa images
            poses: [N, 4, 4]. Camera poses
            hwf: [3,]. Camera intrinsics. h stands for image height, w for image
                 width and f for focal length
        """
        scene = self.scene
        root = self.root
        path = os.path.join('..', 'data', 'synthetic', scene)
        # load JSON file
        with open(os.path.join(path, f'transforms_{root}.json'), 'r') as f:
            meta = json.load(f) # metadata

        # load images and camera poses
        imgs = []
        poses = []
        for frame in meta['frames']:
            fname = os.path.join(path, frame['file_path'] + '.png')
            imgs.append(iio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        # convert to numpy arrays
        imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)
        poses = np.stack(poses, axis=0).astype(np.float32)

        # compute image height, width and camera's focal length
        H, W = imgs.shape[1:3]
        fov_x = meta['camera_angle_x'] # field of view along camera x-axis
        focal = 0.5 * W / np.tan(0.5 * fov_x)
        hwf = np.array([H, W, np.array(focal)])

        imgs = torch.from_numpy(imgs)
        poses = torch.from_numpy(poses)
        hwf = torch.from_numpy(hwf)

        # scale images and camera intrinsics if applicable
        if self.factor is not None:
            imgs, hwf = self.__scale(imgs, hwf)

        return imgs, poses, hwf
