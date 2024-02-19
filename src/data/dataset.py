# standard library modules
import json
import os
from typing import Tuple, List, Union, Callable

# third-party modules
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur, Resize

# custom modules
from utils import utilities as U


# translate across world's z-axis
trans_t = lambda t: torch.Tensor([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., t],
                                 [0., 0., 0., 1.]]).float() 

# rotate around world's x-axis
rot_theta = lambda theta: torch.Tensor([[1., 0., 0., 0.],
                                        [0., np.cos(theta), -np.sin(theta), 0.],
                                        [0., np.sin(theta), np.cos(theta), 0.],
                                        [0., 0., 0., 1.]]).float()

# rotate around world's z-axis
rot_phi = lambda phi: torch.Tensor([[np.cos(phi), -np.sin(phi), 0., 0.],
                                    [np.sin(phi), np.cos(phi), 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]]).float()

def pose_from_spherical(
    radius: float, 
    theta: float,
    phi: float
) -> torch.Tensor:
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
    pose = rot_theta(theta/180. * np.pi) @ pose
    pose = rot_phi(phi/180. * np.pi) @ pose 
    
    return pose


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
        super(SyntheticRealistic).__init__()  # inherit from Dataset
        self.scene = scene
        self.split = split
        self.near = 2.0
        self.far = 8.0
        self.img_mode = img_mode

        imgs, poses, hwf = self.__load() # load imgs, poses and intrinsics
        self.__build_path() # build path to render sample video
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
        x = x[x[:, -1] > 0] # remove poses with negative z-coordinates
        kmeans = KMeans(n_clusters=n_imgs,  n_init=10).fit(x) # kmeans model
        labels = kmeans.labels_
        # compute distances to cluster centers
        dists = np.linalg.norm(x - kmeans.cluster_centers_[labels], axis=1)
        # choose the closest view for every cluster center
        idxs = np.empty((n_imgs,), dtype=int) # array for indices of views
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

    def __build_path(
            self,
            radius: float = 4.0311289,
            theta: float = 50.,
            frames: int = 90
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
        path_poses = [pose_from_spherical(radius, theta, phi)
                        for phi in np.linspace(0, 360, frames, endpoint=False)]
        self.path_poses = torch.stack(path_poses, 0)

class LLFF(Dataset):
    """
    Local Light Field Fusion dataset.
    ----------------------------------------------------------------------------
    """
    @staticmethod
    def __normalize(v: Tensor) -> Tensor:
        """
        Normalizes a vector.
        ------------------------------------------------------------------------
        Args:
            v (Tensor): [N,]. Vector
        Returns:
            v (Tensor): [N,]. Normalized vector
        """
        return v / np.linalg.norm(v)

    @staticmethod
    def __viewmatrix(z: ndarray, up: ndarray, pos: ndarray) -> ndarray:
        """
        Computes the view matrix.
        ------------------------------------------------------------------------
        Args:
            z (ndarray): [3,]. View direction
            up (ndarray): [3,]. Up direction
            pos (ndarray): [3,]. Camera position
        Returns:
            view (ndarray): [3, 4]. View matrix without bottom row
        """
        z = LLFF.__normalize(z)
        y = up
        x = LLFF.__normalize(np.cross(y, z))
        y = LLFF.__normalize(np.cross(z, x))
        matrix = np.stack([x, y, z, pos], axis=1)

        return matrix

    @staticmethod
    def __avg_pose(poses: ndarray) -> ndarray:
        """
        Computes camera to world matrix.
        ------------------------------------------------------------------------
        Args:
            poses (Tensor): [N, 3, 5]. Camera poses
        Returns:
            c2w (Tensor): [N, 3, 5]. Camera to world matrix
        """
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        viewdir = LLFF.__normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([LLFF.__viewmatrix(viewdir, up, center), hwf], 1)

        return c2w

    @staticmethod
    def __recenter_poses(poses: ndarray) -> ndarray:
        """
        Re-centers camera poses.
        ------------------------------------------------------------------------
        Args:
            poses (Tensor): [N, 3, 5]. Camera poses
        Returns:
            poses (Tensor): [N, 3, 5]. Re-centered camera poses
        """
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4]) # last row of camera matrix
        c2w = LLFF.__avg_pose(poses) # average pose
        c2w = np.concatenate([c2w[:3, :4], bottom], axis=-2) # center to world
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2) # camera to world
        
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_

        return poses

    @staticmethod
    def __load(
            basedir,
            factor
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Loads the dataset. It loads images, camera poses, camera bounds and 
        intrinsics.
        ------------------------------------------------------------------------
        Args:
            basedir (str): base directory
            factor (int): resize factor
        Returns:
            imgs (ndarray): [N, H, W, 3]. RGB images
            poses (ndarray): [N, 4, 4]. Camera poses
            bds (ndarray): [N, 2]. Camera bounds
            hwf (ndarray): [3,]. Camera intrinsics. Contains height, width and
                          focal length
        """
        # load camera poses and bounds
        data = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = data[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bounds = data[:, -2:].transpose([1, 0])

        # search for downsampled images
        suffix = '' # path suffix
        if factor > 1:
            suffix = f'_{factor}'
        img_dir = os.path.join(basedir, 'images' + suffix)
        assert os.path.exists(img_dir), f"Images path '{img_dir}' does not exist"

        # load images
        paths = [os.path.join(img_dir, f)
                 for f in sorted(os.listdir(img_dir))
                 if f.endswith(('JPG', 'jpg', 'png'))]
        assert len(paths) == poses.shape[-1], \
                'Mismath between the number of images and poses'
        imgs = np.stack([iio.imread(p)[..., :3] / 255. for p in paths], axis=0)

        # modify camera poses
        H, W, _ = iio.imread(paths[0]).shape
        poses[:2, 4, :] = np.array([H, W]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / factor
        # correct poses ordering
        poses = np.concatenate(
                [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]],
                axis=1
        )
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)
        
        return imgs, poses, bounds

    def __init__(
            self,
            scene: str,
            split: str,
            n_imgs: int,
            white_bkgd: bool = False,
            img_mode: bool = False,
            factor: int = 4,
            bd_factor: float = 0.75,
            recenter: bool = True,
            ndc: bool = True,
    ) -> None:
        """
        Initialize dataset.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene name
            n_imgs (int): number of images
            split (str): split name train/val/test
            white_bkgd (bool): if True, it uses white background
            img_mode (bool): if True, it returns images instead of rays
            factor (int): resize factor
            bd_factor (float): bounding box factor
            recenter (bool): if True, it re-centers the poses
            ndc (bool): if True, use normalized device coordinates
        """
        super(LLFF, self).__init__()
        self.ndc = ndc
        self.img_mode = img_mode
        basedir = os.path.join('..', 'datasets', 'llff', scene)
        imgs, poses, bounds = LLFF.__load(basedir, factor)

        # rescale bounds and poses
        scale = 1. if bd_factor is None else 1. / (bounds.min() * bd_factor)
        poses[..., :3, 3] *= scale
        bounds *= scale

        if recenter:
            poses = LLFF.__recenter_poses(poses)

        c2w = LLFF.__avg_pose(poses)
        self.__build_path(poses, bounds) # build path for video sample
        dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
        i_test = np.argmin(dists)

        imgs = imgs.astype(np.float32)
        hwf = poses[0, :3, -1].astype(np.float32)
        poses = poses[:, :3, :4].astype(np.float32)

        # to tensors
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.poses = torch.tensor(poses, dtype=torch.float32)
        self.hwf = torch.tensor(hwf, dtype=torch.float32)

        self.testimg = self.imgs[i_test]
        self.testpose = self.poses[i_test]

        # define bounds
        if not ndc:
            self.near = bounds.min() * .9
            self.far = bounds.max() * 1.
        else:
            self.near = 0.
            self.far = 1.


        # define rays
        if not self.img_mode:
            # split images into individual per-ray samples
            self.__build_data()

    def __build_data(self) -> None:
        """
        Builds rays and samples.
        ------------------------------------------------------------------------
        """
        H, W, f = self.hwf
        H, W = int(H), int(W)
        self.rgb = self.imgs.reshape(-1, 3) # reshape to pixels
        # get rays
        rays = torch.stack([torch.cat(U.get_rays(H, W, f, p), -1)
                            for p in self.poses], 0)
        rays = rays.reshape(-1, 6)
        rays_o = rays[:, :3] # ray origins
        rays_d = rays[:, 3:] # ray directions

        # map to ndc if necessary
        if self.ndc:
            rays_o, rays_d = U.to_ndc(rays_o, rays_d, 1., self.hwf)
            min_roi = torch.vstack(
                    [rays_o.min(dim=0)[0], 
                    (rays_o + rays_d).min(dim=0)[0]]).min(dim=0)[0]
            max_roi = torch.vstack(
                    [rays_o.max(dim=0)[0], 
                    (rays_o + rays_d).max(dim=0)[0]]).max(dim=0)[0]
            aabb = torch.hstack([min_roi, max_roi])
        else:
            aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])

        self.aabb = aabb
        self.rays_o = rays_o
        self.rays_d = rays_d

    def __build_path(
            self,
            poses: ndarray,
            bounds: ndarray,
            n_views: int = 120,
            n_rots: int = 2,
            zrate: float = 0.5,
            path_zflat: bool = False
    ) -> None:
        """
        Build spiral path for rendering sample video.
        ------------------------------------------------------------------------
        Args:
            poses (ndarray): [..., 4, 4]. Camera poses
            bounds (ndarray): [..., 2]. Near and far bounds
            n_views (int): number of views
            n_rots (int): number of rotations
            zrate (float): rate of z-rotation
            path_zflat (bool): if True, the path is z-flat
        """
        c2w = LLFF.__avg_pose(poses) # get avg pose
        up = LLFF.__normalize(poses[:, :3, 1].sum(0))  # average up
        # compute reasonable focus depth for the scene
        close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
        dt = 0.75
        mean_dz = 1./(((1. - dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # compute radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)

        if path_zflat:
            zloc = -close_depth * .1
            c2w[:3, 3] = c2w[:3, 3] + zloc * c2w[:3, 2]
            rads[2] = 0.
            n_rots = 1
            n_views /= 2
        
        # compute spiral path
        path_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w[:,4:5]

        for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(
                    c2w[:3, :4], 
                    np.array([
                        np.cos(theta), 
                        -np.sin(theta), 
                        -np.sin(theta * zrate), 
                        np.cos(theta)
                    ]) * rads
            )
            z = LLFF.__normalize(c - np.dot(
                    c2w[:3, :4], 
                    np.array([0, 0, -focal, 1.])
                )
            )
            path_poses.append(np.concatenate(
                    [LLFF.__viewmatrix(z, up, c), hwf], 
                    1
                )
            )
            # cast to tensor
            self.path_poses = torch.tensor(path_poses, dtype=torch.float32)


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

    def __len__(self) -> int:
        """Returns the number of training samples."""
        if self.img_mode:
            return self.imgs.shape[0]

        return self.rays_o.shape[0]
