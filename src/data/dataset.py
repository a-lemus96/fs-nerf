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
    def __init__(
            self, 
            scene: str, 
            split: str,
            n_imgs: int = None,
            factor: float = None,
            white_bkgd: bool = False
    ) -> None:
        """
        Initialize the dataset.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene name
            split (str): train, val or test split
            n_imgs (int): number of training images
            factor (float): factor to scale images and camera intrinsics
        Returns:
            None
        """
        super(SyntheticRealistic).__init__()  # inherit from Dataset
        self.scene = scene
        self.split = split
        self.factor = factor
        self.near = 2.0
        self.far = 6.0

        # load the dataset
        imgs, depths, poses, hwf = self.__load()
        self.hwf = hwf
        # compute background color
        if white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]
        # choose random index for test image, pose and depth map
        idx = np.random.randint(0, imgs.shape[0])
        self.testimg = imgs[idx]
        self.testpose = poses[idx]
        self.testdepth = depths[idx]
        self.testdepth[torch.isinf(self.testdepth)] = 0.

        # draw a random sample of indices
        if n_imgs is not None:
            idxs = np.random.choice(imgs.shape[0], n_imgs, replace=False)
            imgs = imgs[idxs]
            depths = depths[idxs]
            poses = poses[idxs]

        # compute rays
        H, W, f = hwf
        rays = torch.stack([torch.cat(U.get_rays(H, W, f, p), -1) 
                            for p in poses], 0)
        rays = rays.reshape(-1, 6)
        self.rays_o = rays[:, :3]
        self.rays_d = rays[:, 3:]

        # add pixel colors and depth values
        self.rgb = imgs.reshape(-1, 3)
        self.depth = depths.reshape(-1)


    def __len__(self) -> int:
        """Compute the number of training samples.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            N (int): number of training samples
        """
        return self.rgb.shape[0]


    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get a training sample by index.
        ------------------------------------------------------------------------
        Args:
            idx (int): index of the training sample
        Returns:
            ray_o (Tensor): [3,]. Ray origin
            ray_d (Tensor): [3,]. Ray direction
            rgb (Tensor): [3,]. Pixel RGB color
            depth (Tensor): [1,]. Pixel depth value
        """
        return self.rays_o[idx], self.rays_d[idx], self.rgb[idx], self.depth[idx]


    def __factor(
            self, 
            imgs: Tensor, 
            depths: Tensor,
            hwf: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Downsample images and apply resize factor to camera intrinsics.
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            depths (Tensor): [N, H, W]. Depth maps
            hwf (Tensor): [3,]. Camera intrinsics
        Returns:
            new_imgs (Tensor): [N, H // factor, W // factor, 4]. RGBa images
            new_depths (Tensor): [N, H // factor, W // factor]. Depth maps
            new_hwf (Tensor): [3,]. Camera intrinsics
        """
        # apply factor to camera intrinsics
        H, W, f = hwf
        new_W, new_H = H // factor, W // factor
        new_focal = hwf[2] / float(factor)
        new_hwf = torch.Tensor((new_H, new_W, new_focal))

        # downsample images
        new_imgs = Resize((new_H, new_W))(imgs)
        new_depths = Resize((new_H, new_W))(depths)

        return new_imgs, new_depths, new_hwf

    def __ray_depth(self, depths: Tensor, hwf: Tensor) -> Tensor:
        """Given a set of depth maps using z-coords, compute depth values along
        rays.
        ------------------------------------------------------------------------
        Args:
            depths (Tensor): [N, H, W]. Depth maps
            hwf (Tensor): [3,]. Camera intrinsics
        Returns:
            ray_depths (Tensor): [N, H, W]. Depth values along rays
        """
        H, W, f = hwf
        H, W = int(H), int(W)
        rays_d = U.get_rays(H, W, f) # get local ray directions

        # compute depth values along rays
        rays_d = rays_d[..., -1]
        t_depths = -depths / rays_d[None, ...]

        return t_depths

    def __load(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Loads the dataset. It loads images, camera poses,  camera intrinsics and
        depth maps.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            depths (Tensor): [N, H, W]. Depth maps (along z axis)
            poses (Tensor): [N, 4, 4]. Camera poses
            hwf (Tensor): [3,]. Camera intrinsics. It contains height, width and
                          focal length
        """
        scene = self.scene
        path = os.path.join('..', 'datasets', 'synthetic', scene)
        # load JSON file
        with open(os.path.join(path, f'transforms_train.json'), 'r') as f:
            meta = json.load(f) # metadata

        # get training, validation and test indices
        train_idxs = np.arange(0, 200, 2) # training indices
        mask = np.isin(np.arange(0, 200), train_idxs) # mask for test
        test_idxs = np.arange(0, 200) # test indices
        test_idxs = np.random.choice(test_idxs[~mask], 50, False) # indices
        mask2 = mask + np.isin(np.arange(0, 200), test_idxs) # mask for val
        vals_idxs = np.arange(0, 200) # validation indices
        val_idxs = vals_idxs[~mask2] # indices

        # create dictionary with indices
        idxs = {'train': train_idxs, 'val': val_idxs, 'test': test_idxs}

        # load images and camera poses
        imgs = []
        disps = []
        poses = []
        depth_str = '_depth_0001.png'
        for i, frame in enumerate(meta['frames']):
            if i in idxs[self.split]:
                # camera pose
                poses.append(np.array(frame['transform_matrix']))
                # frame image and depth map
                fname = os.path.join(path, frame['file_path'] + '.png')
                imgs.append(iio.imread(fname)) # RGBa image
                fname = os.path.join(path, frame['file_path'] + depth_str)
                disps.append(iio.imread(fname)) # disparity map

        # convert to numpy arrays
        poses = np.stack(poses, axis=0).astype(np.float32)
        imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)
        disps = (np.stack(disps, axis=0) / 255.).astype(np.float32)
        depths = (1. - disps) * 8. # apply inverse affine transformation
        depths[depths == 8.] = np.inf

        # compute image height, width and camera's focal length
        H, W = imgs.shape[1:3]
        fov_x = meta['camera_angle_x'] # field of view along camera x-axis
        focal = 0.5 * W / np.tan(0.5 * fov_x)
        hwf = np.array([H, W, np.array(focal)])

        imgs = torch.from_numpy(imgs)
        depths = torch.from_numpy(depths[..., 0])
        # convert z depth to along-ray depth
        depths = self.__ray_depth(depths, hwf)
        poses = torch.from_numpy(poses)
        hwf = torch.from_numpy(hwf)

        # scale images and camera intrinsics if applicable
        if self.factor is not None:
            imgs, depths, hwf = self.__scale(imgs, depths, hwf)

        return imgs, depths, poses, hwf
