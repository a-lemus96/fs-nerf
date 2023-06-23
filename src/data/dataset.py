# standard library modules
import json
import os
from typing import Tuple, List, Union, Callable

# third-party modules
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Resize

# custom modules
import utils.utilities as U

class SyntheticRealistic(Dataset):
    """
    Synthetic realistic dataset. It is made up of N x H x W ray origins and
    directions in world coordinate frame paired with ground truth pixel colors
    and depth values. The dataset is stored in a directory named 'synthetic'.
    Here, N is the number of training images of size H x W.
    ----------------------------------------------------------------------------
    """
    def __init__(self, 
            scene: str, 
            root: str, 
            factor: float = None,
            white_bkgd: bool = False) -> None:
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
        self.near = 2.0
        self.far = 6.0

        # load the dataset
        imgs, poses, hwf = self.__load()
        self.hwf = hwf
        # compute background color
        if white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]
        # choose random index for test image and pose
        idx = np.random.randint(0, imgs.shape[0])
        self.testimg = imgs[idx]
        self.testpose = poses[idx]

        # compute rays
        H, W, f = hwf
        rays = torch.stack([torch.cat(U.get_rays(H, W, f, p), -1) 
                            for p in poses], 0)
        rays = rays.reshape(-1, 6)
        self.rays_o = rays[:, :3]
        self.rays_d = rays[:, 3:]

        # add pixel colors
        self.rgb = imgs.reshape(-1, 3)


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
            rgb (Tensor): [3,]. Pixel RGB color
        """
        return self.rays_o[idx], self.rays_d[idx], self.rgb[idx]


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
        Loads the dataset. It loads images, camera poses,  camera intrinsics and
        depth maps.
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
        path = os.path.join('..', 'datasets', 'synthetic', scene)
        # load JSON file
        with open(os.path.join(path, f'transforms_{root}.json'), 'r') as f:
            meta = json.load(f) # metadata

        # load images and camera poses
        imgs = []
        disps = []
        poses = []
        for frame in meta['frames']:
            poses.append(np.array(frame['transform_matrix'])) # camera pose
            fname = os.path.join(path, frame['file_path'] + '.png')
            imgs.append(iio.imread(fname)) # RGBa image
            fname = os.path.join(path, frame['file_path'] + '_depth_0001.png')
            print(iio.imread(fname).shape)
            disps.append(iio.imread(fname)) # disparity map

        # convert to numpy arrays
        poses = np.stack(poses, axis=0).astype(np.float32)
        imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)
        disps = np.stack(disps, axis=0).astype(np.float32)

        # convert disparity maps into depth maps
        depths = 1./disps
        depths[depths == np.inf]
        print(depths[0, ..., 0])

        plt.imshow(depths[0, ..., 0])
        plt.savefig('depth0.png')
        plt.imshow(depths[0, ..., 1])
        plt.savefig('depth1.png')
        # check if both depth maps are equal
        if not np.allclose(depths[..., 0], depths[..., 1]):
            print('Depth maps are not equal')
        exit()

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
