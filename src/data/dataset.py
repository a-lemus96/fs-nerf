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
        imgs, depths, poses, hwf = self.__load()
        self.hwf = hwf
        # compute background color
        if white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]

        # choose random index for display image, pose and depth map
        idx = np.random.randint(0, imgs.shape[0])
        self.testimg = imgs[idx]
        self.testpose = poses[idx]
        self.testdepth = depths[idx]

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
        self.depths = depths[idxs]
        self.poses = poses[idxs]
        
        # create training samples
        self.__build_samples(self.imgs, self.depths, self.poses, self.hwf)

    def __len__(self) -> int:
        """Compute the number of training samples.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            N (int): number of training samples
        """
        if self.img_mode:
            return self.imgs.shape[0]

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
        if self.img_mode:
            return self.imgs[idx], self.depths[idx]

        return self.rays_o[idx], self.rays_d[idx], self.rgb[idx], self.depth[idx]

    def __build_samples(
            self,
            imgs: Tensor,
            depths: Tensor,
            poses: Tensor,
            hwf: Tensor
    ) -> None:
        """
        Build set of rays in world coordinate frame and their corresponding 
        pixel RGB colors and depth values.
        ------------------------------------------------------------------------
        Args:
            
        """
        # compute ray origins and directions
        H, W, f = hwf
        rays = torch.stack([torch.cat(U.get_rays(H, W, f, p), -1) 
                            for p in poses], 0)
        rays = rays.reshape(-1, 6)
        self.rays_o = rays[:, :3]
        self.rays_d = rays[:, 3:]

        # add pixel colors and depth values
        self.rgb = imgs.reshape(-1, 3)
        self.depth = depths.reshape(-1)

    def __downsample(
            self, 
            imgs: Tensor, 
            depths: Tensor,
            hwf: Tensor,
            factor: int
    ) -> None:
        """
        Downsample images and apply resize factor to camera intrinsics.
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            depths (Tensor): [N, H, W]. Depth maps
            hwf (Tensor): [3,]. Camera intrinsics
            factor (int): resize factor
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
        with open(os.path.join(path, f'transforms_{self.split}.json'), 'r') as f:
            meta = json.load(f) # metadata

        # load images and camera poses
        imgs = []
        depths = []
        poses = []
        depth_str = '_depth_0001.png' # depth map end of file name
        for i, frame in enumerate(meta['frames']):
            # camera pose
            poses.append(np.array(frame['transform_matrix']))
            # frame image and depth map
            fname = os.path.join(path, frame['file_path'] + '.png')
            imgs.append(iio.imread(fname)) # RGBa image
            fname = os.path.join(path, frame['file_path'] + depth_str)
            depths.append(iio.imread(fname)) # depth map

        # convert to numpy arrays
        poses = np.stack(poses, axis=0).astype(np.float32)
        imgs = (np.stack(imgs, axis=0) / 255.).astype(np.float32)
        depths = (np.stack(depths, axis=0) / 255.).astype(np.float32)
        depths = (1. - depths) * 8. # apply inverse affine transformation
        depths[depths == 8.] = np.inf

        # compute image height, width and camera's focal length
        H, W = imgs.shape[1:3]
        fov_x = meta['camera_angle_x'] # field of view along camera x-axis
        focal = 0.5 * W / np.tan(0.5 * fov_x)
        hwf = np.array([H, W, np.array(focal)])

        # create tensors
        poses = torch.from_numpy(poses)
        imgs = torch.from_numpy(imgs)
        depths = torch.from_numpy(depths[..., 0])
        # convert z depth to along-ray depth
        depths = self.__ray_depth(depths, hwf)
        hwf = torch.from_numpy(hwf)

        return imgs, depths, poses, hwf


    def gaussian_downsample(self, t: int) -> None:
        """
        Applies Gaussian blur + downsampling to images and depth maps.
        ------------------------------------------------------------------------
        Args:
            t (int): Gaussian blur standard deviation
        Returns:
            None
        """    
        t = int(t)
        if t > 0:
            # permute images and depths to [N, C, H, W] format
            imgs = torch.permute(self.imgs, (0, 3, 1, 2)) # [N, 3, H, W]
            depths = torch.unsqueeze(self.depths, 1) # [N, 1, H, W]

            # apply Gaussian blur
            blur = GaussianBlur(6 * t + 1, sigma=float(t))
            imgs = blur(imgs) # [N, 3, H, W]
            depths = blur(depths) # [N, 1, H, W]

            # permute images and depths to [N, H, W, C] format
            imgs = torch.permute(imgs, (0, 2, 3, 1)) # [N, H, W, 3]
            depths = torch.squeeze(depths, 1) # [N, H, W]

            # downsample images and depths
            imgs, depths, hwf = self.__downsample(imgs, depths, self.hwf, t//2)

            # re-build training samples
            self.__build_samples(imgs, depths, self.poses, hwf)
