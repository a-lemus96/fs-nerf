# standard library modules
import json
import os
from typing import Tuple
from dataclasses import dataclass

# third-party modules
import imageio as iio
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Resize

# custom modules
from utils import utilities as U


# translate across world's z-axis
trans_t = lambda t: torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, t],
        [0.0, 0.0, 0.0, 1.0],
    ]
).float()

# rotate around world's x-axis
rot_theta = lambda theta: torch.Tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta), 0.0],
        [0.0, np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
).float()

# rotate around world's z-axis
rot_phi = lambda phi: torch.Tensor(
    [
        [np.cos(phi), -np.sin(phi), 0.0, 0.0],
        [np.sin(phi), np.cos(phi), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
).float()


def pose_from_spherical(radius: float, theta: float, phi: float) -> torch.Tensor:
    """
    Computes 4x4 camera pose from 3D location expressed in spherical coords.
    Camera frame points toward object with its y-axis tangent to the virtual
    spherical surface defined by the given radius.
    ----------------------------------------------------------------------------
    Args:
        radius: float. Sphere radius
        theta: 0° < float < 90°. colatitude angle
        phi: 0° < float < 360°. Azimutal angle
    Returns:
        pose: [..., 4, 4]. Camera to world transformation
    """

    pose = trans_t(radius)
    pose = rot_theta(theta / 180.0 * np.pi) @ pose
    pose = rot_phi(phi / 180.0 * np.pi) @ pose

    return pose


class BlenderDataset(Dataset):
    """
    Blender dataset. It is made up of N x H x W ray origins and
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
        img_mode: bool = False,
        white_bkgd: bool = False,
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
        super(BlenderDataset).__init__()  # inherit from Dataset
        self.scene = scene
        self.split = split
        self.near = 2.0
        self.far = 6.0
        self.ndc = False
        self.img_mode = img_mode

        imgs, poses, hwf = self.__load()  # load imgs, poses and intrinsics
        self.__build_path()  # build path to render sample video
        self.hwf = hwf

        # compute background color
        if white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]

        # choose random index for visual comparisons
        idx = np.random.randint(0, imgs.shape[0])
        self.testimg = imgs[idx]
        self.testpose = poses[idx]

        # apply K-means to draw N views and ensure maximum scene coverage
        x = poses[:, :3, 3]
        x = x[x[:, -1, -1] > 0]  # remove poses with negative z-coordinates
        kmeans = KMeans(n_clusters=n_imgs, n_init=10).fit(x)  # kmeans model
        labels = kmeans.labels_
        # compute distances to cluster centers
        dists = np.linalg.norm(x - kmeans.cluster_centers_[labels], axis=1)
        # choose the closest view for every cluster center
        idxs = np.empty((n_imgs,), dtype=int)  # array for indices of views
        for i in range(n_imgs):
            cluster_dists = np.where(labels == i, dists, np.inf)
            idxs[i] = np.argmin(cluster_dists)

        self.imgs = imgs[idxs]
        self.poses = poses[idxs]
        # axis-aligned bounding box for occupancy grid estimator
        self.aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])

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
        self, imgs: Tensor, poses: Tensor, hwf: Tuple[int, int, float]
    ) -> None:
        """
        Build set of rays in world coordinate frame and their corresponding
        pixel RGB values.
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            poses (Tensor): [N, 4, 4]. Camera poses
            hwf (Tuple): [3,]. Camera intrinsics
        """
        # compute ray origins and directions
        rays = torch.stack([torch.cat(U.get_rays(p, hwf), -1) for p in poses], 0)
        rays = rays.reshape(-1, 6)
        self.rays_o = rays[:, :3]  # ray origins
        self.rays_d = rays[:, 3:]  # ray directions
        self.rgb = imgs.reshape(-1, 3)  # reshape to [N, 3]

    def __downsample(
        self, imgs: Tensor, factor: int, hwf: Tuple[int, int, float]
    ) -> None:
        """
        Downsample images and apply resize factor to camera intrinsics.
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            factor (int): resize factor
            hwf (Tuple): [3,]. Camera intrinsics
        Returns:
            new_imgs (Tensor): [N, H // factor, W // factor, 4]. RGBa images
            new_hwf (Tuple): [3,]. Camera intrinsics
        """
        # apply factor to camera intrinsics
        H, W, f = hwf
        new_H, new_W = H // factor, W // factor
        new_focal = f / float(factor)
        new_hwf = (new_H, new_W, new_focal)
        # downsample images
        new_imgs = Resize((new_H, new_W))(imgs)

        return new_imgs, new_hwf

    def __load(self) -> Tuple[Tensor, Tensor, Tuple[int, int, float]]:
        """
        Loads the dataset. It loads images, camera poses and intrinsics.
        ------------------------------------------------------------------------
        Args:
            None
        Returns:
            imgs (Tensor): [N, H, W, 4]. RGBa images
            poses (Tensor): [N, 4, 4]. Camera poses
            hwf (Tuple): [3,]. Camera intrinsics
        """
        scene = self.scene
        path = os.path.join("..", "datasets", "synthetic", scene)
        # load JSON file
        with open(os.path.join(path, f"transforms_{self.split}.json"), "r") as f:
            meta = json.load(f)  # metadata

        # load images and camera poses
        imgs = []
        poses = []
        for frame in meta["frames"]:
            # camera pose
            poses.append(np.array(frame["transform_matrix"]))
            # frame image
            fname = os.path.join(path, frame["file_path"] + ".png")
            imgs.append(iio.imread(fname))  # RGBa image

        # convert to numpy arrays
        poses = np.stack(poses, axis=0).astype(np.float32)
        imgs = (np.stack(imgs, axis=0) / 255.0).astype(np.float32)

        # compute image height, width and camera's focal length
        H, W = imgs.shape[1:3]
        fov_x = meta["camera_angle_x"]  # field of view along camera x-axis
        focal = 0.5 * W / np.tan(0.5 * fov_x)
        hwf = (H, W, focal.item())

        # create tensors
        poses = torch.from_numpy(poses)
        imgs = torch.from_numpy(imgs)

        return imgs, poses, hwf

    def __build_path(
        self, radius: float = 4.0311289, theta: float = 50.0, frames: int = 90
    ) -> torch.Tensor:
        """
        Creates a set of frames for inward facing camera poses using constant
        radius and theta, while varying azimutal angle within the interval
        [0,360].
        ------------------------------------------------------------------------
        Args:
            radius: Sphere radius
            theta: 0° < float < 90°. Colatitude angle
            frames: Number of samples along azimutal interval
        """
        path_poses = [
            pose_from_spherical(radius, theta, phi)
            for phi in np.linspace(0, 360, frames, endpoint=False)
        ]
        self.path_poses = torch.stack(path_poses, 0)
