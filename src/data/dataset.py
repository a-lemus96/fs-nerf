# standard library modules
import json
import os
from typing import Tuple, List, Union, Callable

# third-party modules
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur, Resize

# custom modules
import utils.utilities as U

class SyntheticRealistic(Dataset):
    """
    Synthetic realistic dataset. It is made up of N x H x W ray origins and
    directions in world coordinate frame paired with ground truth pixel values. 
    The dataset is stored in a directory named 'synthetic'. Here, N is the 
    number of training images of size H x W.
    ----------------------------------------------------------------------------
    """
    def __init__(
            self, 
            scene: str, 
            split: str,
            n_imgs: int = None,
            white_bkgd: bool = False,
            img_mode: bool = False
    ) -> None:
        """
        Initialize the dataset.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene name
            split (str): train, val or test split
            n_imgs (int): number of training images
            white_bkgd (bool): whether to use white background
            img_mode (bool): wether to iterate over rays or images
        Returns:
            None
        """
        super(SyntheticRealistic).__init__()  # inherit from Dataset
        self.scene = scene
        self.split = split
        self.near = 2.0
        self.far = 8.0
        self.img_mode = img_mode

        # load the dataset
        imgs, poses, hwf = self.__load()
        self.hwf = hwf
        # compute background color
        if white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]

        # choose random index for visual comparisons
        idx = np.random.randint(0, imgs.shape[0])
        self.testimg = imgs[idx]
        self.testpose = poses[idx]

        # apply K-means to draw N views and ensure maximum scene coverage
        x = poses[:, :3, 3]
        kmeans = KMeans(n_clusters=n_imgs,  n_init=10).fit(x) # kmeans model
        labels = kmeans.labels_
        # compute distances to cluster centers
        dists = np.linalg.norm(x - kmeans.cluster_centers_[labels], axis=1)
        # choose the closest view for every cluster center
        idxs = np.empty((n_imgs,), dtype=int) # array for indices of views
        for i in range(n_imgs):
            cluster_dists = np.where(labels == i, dists, np.inf)
            idxs[i] = np.argmin(cluster_dists)

        # full resolution images
        self.imgs = imgs[idxs]
        self.poses = poses[idxs]

        if not self.img_mode:
            # split images into individual per-ray samples
            self.__build_data(self.imgs, self.poses, self.hwf)


    def __len__(self) -> int:
        """Compute the number of training samples.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            N (int): number of training samples
        """
        if self.img_mode:
            return len(self.imgs)

        return len(self.rgb)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get a training sample by index.
        ------------------------------------------------------------------------
        Args:
            idx (int): index of the training sample
        Returns:
            ray_o (Tensor): [3,]. Ray origin
            ray_d (Tensor): [3,]. Ray direction
            rgb (Tensor): [3,]. Pixel RGB color
        """
        if self.img_mode:
            return self.imgs[idx], self.poses[idx]

        return self.rays_o[idx], self.rays_d[idx], self.rgb[idx]

    def __build_data(
            self,
            imgs: Tensor,
            poses: Tensor,
            hwf: Tensor
    ) -> None:
        """
        Build set of rays in world coordinate frame and their corresponding 
        pixel RGB values.
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            poses (Tensor): [N, 4, 4]. Camera poses
            hwf (Tensor): [3,]. Camera intrinsics
        """
        # compute ray origins and directions
        H, W, f = hwf
        rays = torch.stack([torch.cat(U.get_rays(H, W, f, p), -1) 
                            for p in poses], 0)
        rays = rays.reshape(-1, 6)
        self.rays_o = rays[:, :3] # ray origins
        self.rays_d = rays[:, 3:] # ray directions
        self.rgb = imgs.reshape(-1, 3) # reshape to [N, 3]

    def __downsample(
            self, 
            imgs: Tensor, 
            hwf: Tensor,
            factor: int
    ) -> None:
        """
        Downsample images and apply resize factor to camera intrinsics.
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            hwf (Tensor): [3,]. Camera intrinsics
            factor (int): resize factor
        Returns:
            new_imgs (Tensor): [N, H // factor, W // factor, 4]. RGBa images
            new_hwf (Tensor): [3,]. Camera intrinsics
        """
        # apply factor to camera intrinsics
        H, W, f = hwf
        new_H, new_W = int(H) // factor, int(W) // factor
        new_focal = hwf[2] / float(factor)
        new_hwf = torch.Tensor((new_H, new_W, new_focal))
        # downsample images
        new_imgs = Resize((new_H, new_W))(imgs)

        return new_imgs, new_hwf

    def __load(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Loads the dataset. It loads images, camera poses and intrinsics.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            poses (Tensor): [N, 4, 4]. Camera poses
            hwf (Tensor): [3,]. Camera intrinsics. It contains height, width and
                          focal length
        """
        scene = self.scene
        path = os.path.join('..', 'datasets', 'synthetic', scene)
        # load JSON file
        with open(os.path.join(path, f'transforms_{self.split}.json'), 'r') as f:
            meta = json.load(f) # metadata

        # load images and camera poses
        imgs = []
        poses = []
        for frame in meta['frames']:
            # camera pose
            poses.append(np.array(frame['transform_matrix']))
            # frame image
            fname = os.path.join(path, frame['file_path'] + '.png')
            imgs.append(iio.imread(fname)) # RGBa image

        # convert to numpy arrays
        poses = np.stack(poses, axis=0).astype(np.float32)
        imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)

        # compute image height, width and camera's focal length
        H, W = imgs.shape[1:3]
        fov_x = meta['camera_angle_x'] # field of view along camera x-axis
        focal = 0.5 * W / np.tan(0.5 * fov_x)
        hwf = np.array([H, W, np.array(focal)])

        # create tensors
        poses = torch.from_numpy(poses)
        imgs = torch.from_numpy(imgs)
        hwf = torch.from_numpy(hwf)

        return imgs, poses, hwf


    def gaussian_downsample(self, t: int) -> None:
        """
        Applies Gaussian blur + downsampling to images.
        ------------------------------------------------------------------------
        Args:
            t (int): Gaussian blur standard deviation
        Returns:
            None
        """    
        t = int(t)
        if t > 0:
            # permute images to [N, C, H, W] format
            imgs = torch.permute(self.imgs, (0, 3, 1, 2)) # [N, 3, H, W]

            # apply Gaussian blur
            blur = GaussianBlur(6 * t + 1, sigma=float(t))
            imgs = blur(imgs) # [N, 3, H, W]

            # downsample images
            imgs, hwf = self.__downsample(imgs, self.hwf, 1)
            # permute images back to [N, H, W, C] format
            imgs = torch.permute(imgs, (0, 2, 3, 1)) # [N, H, W, 3]

            # re-build training samples
            self.__build_data(imgs, self.poses, hwf)

            return imgs, hwf

        return self.imgs, self.hwf
