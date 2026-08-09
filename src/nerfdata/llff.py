# standard library modules
import os
from typing import Tuple

# third-party modules
import imageio as iio
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset

# custom modules
from utils import utilities as U


class LLFFDataset(Dataset):
    """
    Represents an instance of a Local Light Field Fusion dataset.
    ----------------------------------------------------------------------------
    """

    LLFF_HOLD = 8  # every LLFF_HOLD-th view is held out for testing

    def __init__(
        self,
        scene: str,
        img_mode: bool = False,
        ndc: bool = True,
    ) -> None:
        """
        Loads the scene: images, poses, bounds and intrinsics.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene folder name under ../datasets/llff/
            img_mode (bool): if True, the dataset yields whole images instead
                             of rays
            ndc (bool): if True, maps rays to normalized device coordinates
        """
        super(LLFFDataset, self).__init__()
        (
            img_paths,
            self.poses,
            self.hwf,
            self.min_bound,
            self.max_bound,
            self.path_poses,
        ) = LLFFDataset.load_scene(scene)
        self.poses = torch.tensor(self.poses, dtype=torch.float32)
        self.img_mode = img_mode
        self.ndc = ndc

        imgs = LLFFDataset.load_img_files(img_paths)
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.__set_bounds()
        if not self.img_mode:
            self.__build_samples()

    @classmethod
    def __from_split(
        cls,
        imgs: Tensor,
        poses: Tensor,
        min_bound: float,
        max_bound: float,
        hwf: Tuple[int, int, float],
        path_poses: ndarray,
        img_mode: bool,
        ndc: bool,
    ) -> "LLFFDataset":
        """
        Builds a dataset instance from in-memory tensors, bypassing disk
        access. Used internally by split().
        ------------------------------------------------------------------------
        Args:
            imgs (Tensor): [N, H, W, 3]. RGB images scaled to [0, 1]
            poses (Tensor): [N, 3, 4]. Camera poses
            min_bound (float): minimum value across the poses
            max_bound (float): maximum value across the poses
            hwf (Tuple[int, int, float]): image height, width and focal length
            path_poses (ndarray): [M, 3, 4]. Spiral path poses for rendering
                                  sample videos
            img_mode (bool): if True, the dataset yields whole images instead
                             of rays
            ndc (bool): if True, maps rays to normalized device coordinates
        Returns:
            dataset (LLFFDataset): dataset over the given views
        """
        self = cls.__new__(cls)
        Dataset.__init__(self)
        self.imgs = imgs
        self.poses = poses
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.hwf = hwf
        self.path_poses = path_poses
        self.img_mode = img_mode
        self.ndc = ndc
        self.__set_bounds()
        if not self.img_mode:
            self.__build_samples()

        return self

    def __set_bounds(self) -> None:
        """
        Defines the ray bounds from the scene's min/max poses bounds.
        ------------------------------------------------------------------------
        """
        if not self.ndc:
            self.near = self.min_bound * 0.9
            self.far = self.max_bound * 1.0
        else:
            self.near = 0.0
            self.far = 1.0

    def __build_samples(self) -> None:
        """
        Builds rays and samples.
        ------------------------------------------------------------------------
        """
        H, W, _ = self.hwf
        self.rgb = self.imgs.reshape(-1, 3)  # reshape to pixels
        # get rays
        rays = torch.stack(
            [torch.cat(U.get_rays(p, self.hwf), -1) for p in self.poses], 0
        )
        rays = rays.reshape(-1, 6)
        rays_o = rays[:, :3]  # ray origins
        rays_d = rays[:, 3:]  # ray directions

        # map to ndc if necessary
        if self.ndc:
            rays_o, rays_d = U.to_ndc(rays_o, rays_d, self.hwf, 1.0)
            min_roi = torch.vstack(
                [rays_o.min(dim=0)[0], (rays_o + rays_d).min(dim=0)[0]]
            ).min(dim=0)[0]
            max_roi = torch.vstack(
                [rays_o.max(dim=0)[0], (rays_o + rays_d).max(dim=0)[0]]
            ).max(dim=0)[0]
            aabb = torch.hstack([min_roi, max_roi])
            aabb = aabb / 2 ** (4 - 1)
        else:
            aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])

        self.aabb = aabb
        self.rays_o = rays_o
        self.rays_d = rays_d

    def split(
        self, n_views: int = -1, train_img_mode: bool = False
    ) -> Tuple["LLFFDataset", "LLFFDataset"]:
        """
        Splits this dataset into train/test datasets following the FreeNeRF
        protocol. Reuses the already-loaded images and poses instead of
        re-reading them from disk, so the parent dataset can be discarded
        right after this call.
        ------------------------------------------------------------------------
        - Test split: every LLFF_HOLD-th view (indices 0, 8, 16, ...). It is
          evaluated in full and is currently reused as the validation split
          by the caller.
        - Train pool: the remaining views.
        - Few-shot train split: `n_views` evenly spaced picks over the train
          pool. LLFF captures sweep the scene sequentially, so evenly spaced
          indices translate to evenly spaced camera poses. Passing a
          negative `n_views` keeps the whole pool.
        Args:
            n_views (int): number of training views to keep. A negative
                           value keeps every view left in the train pool
            train_img_mode (bool): if True, the training dataset yields whole
                                   images instead of rays. The test dataset is
                                   always built in image mode
        Returns:
            train_dataset (LLFFDataset): dataset over the training views
            test_dataset (LLFFDataset): dataset over the held-out views
        """
        all_indices = np.arange(len(self.poses))
        test_ids = all_indices[all_indices % LLFFDataset.LLFF_HOLD == 0]
        pool = all_indices[all_indices % LLFFDataset.LLFF_HOLD != 0]

        if n_views < 0:
            train_ids = pool
        else:
            assert 0 < n_views <= len(pool), (
                f"ValueError, the number of training views must be in [1, {len(pool)}], "
                f"got {n_views}."
            )
            idx_sub = [round(i) for i in np.linspace(0, len(pool) - 1, n_views)]
            train_ids = pool[idx_sub]

        test_dataset = LLFFDataset.__from_split(
            self.imgs[test_ids],
            self.poses[test_ids],
            self.min_bound,
            self.max_bound,
            self.hwf,
            self.path_poses,
            True,
            self.ndc,
        )
        train_dataset = LLFFDataset.__from_split(
            self.imgs[train_ids],
            self.poses[train_ids],
            self.min_bound,
            self.max_bound,
            self.hwf,
            self.path_poses,
            train_img_mode,
            self.ndc,
        )

        return train_dataset, test_dataset

    def to(self, device: torch.device) -> "LLFFDataset":
        """
        Moves dataset tensors to the given device in-place, loading only what
        is needed for the current mode to avoid duplicating data on the GPU.

        In ray mode (img_mode=False): moves rays_o, rays_d, and rgb.
        In image mode (img_mode=True): moves imgs only.
        poses is always moved as it is small (N x 3 x 4).
        ------------------------------------------------------------------------
        Args:
            device (torch.device): target device
        Returns:
            self
        """
        self.poses = self.poses.to(device)
        if self.img_mode:
            self.imgs = self.imgs.to(device)
        else:
            self.rays_o = self.rays_o.to(device)
            self.rays_d = self.rays_d.to(device)
            self.rgb = self.rgb.to(device)
        return self

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
        if self.img_mode:
            return self.imgs[idx], self.poses[idx]

        return self.rays_o[idx], self.rays_d[idx], self.rgb[idx]

    def __len__(self) -> int:
        """Returns the number of training samples."""
        if self.img_mode:
            return self.imgs.shape[0]

        return self.rays_o.shape[0]

    # ------------------------------------------------------------------------
    # Dataset loading logic
    # ------------------------------------------------------------------------

    @staticmethod
    def load_img_files(img_paths: ndarray) -> ndarray:
        """
        Reads image files into a normalized RGB array.
        ------------------------------------------------------------------------
        Args:
            img_paths (ndarray): [N,]. Image file paths
        Returns:
            imgs (ndarray): [N, H, W, 3]. RGB images scaled to [0, 1]
        """
        imgs = np.stack([iio.imread(p)[..., :3] / 255.0 for p in img_paths], axis=0)

        return imgs

    @classmethod
    def load_scene(
        cls,
        scene: str,
        bd_factor: float = 0.75,
        recenter: bool = True,
    ) -> Tuple[ndarray, ndarray, Tuple[int, int, float], float, float, ndarray]:
        """
        Loads image paths, camera poses, bounds and intrinsics from an llff
        dataset folder.
        ------------------------------------------------------------------------
        Expected scene folder structure:
            images_8/          -> contains image files
            poses_bounds.npy   -> poses file
        Args:
            scene (str): scene folder name under ../datasets/llff/
            bd_factor (float): shrinks the scene so the nearest bound sits at
                               1 / bd_factor. None leaves the scale untouched
            recenter (bool): if True, expresses poses relative to the average pose
        Returns:
            img_paths (ndarray): [N,]. Absolute paths to the scene images
            poses (ndarray): [N, 3, 4]. Camera poses
            hwf (Tuple[int, int, float]): image height, width and focal length
            min_bound (float): minimum value across the poses
            max_bound (float): maximum value across the poses
            path_poses (ndarray): [M, 3, 4]. Spiral path poses for rendering
                                  sample videos
        """
        base_folder_path = os.path.normpath("../datasets/llff/")
        assert os.path.isdir(
            base_folder_path
        ), f"LLFF dataset folder {os.path.abspath(base_folder_path)} not found."
        all_scenes = os.listdir(path=base_folder_path)
        assert (
            scene in all_scenes
        ), f"Scene '{scene}' not found in local LLFF dataset folder."

        # load camera poses and bounds
        base_scene_folder_path = os.path.join(base_folder_path, scene)
        data = np.load(os.path.join(base_scene_folder_path, "poses_bounds.npy"))
        poses = data[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bounds = data[:, -2:].transpose([1, 0])

        # load the downsampled images
        imgs_folder_path = os.path.normpath(
            os.path.join(base_folder_path, scene, "images_8/")
        )
        assert os.path.isdir(
            imgs_folder_path
        ), f"Images folder path {os.path.abspath(imgs_folder_path)} not found."
        img_paths = [
            os.path.abspath(os.path.join(imgs_folder_path, f))
            for f in sorted(os.listdir(imgs_folder_path))
            if f.endswith(("JPG", "jpg", "png"))
        ]
        img_paths = np.array(img_paths)
        assert (
            len(img_paths) == poses.shape[-1]
        ), "Mismath between the number of images and poses"

        # modify camera poses
        H, W, _ = iio.imread(img_paths[0]).shape
        poses[:2, 4, :] = np.array([H, W]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / 8.0
        # correct poses ordering
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], axis=1
        )

        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)

        (poses, hwf, min_bound, max_bound, path_poses) = cls.__postprocess_poses(
            poses, bounds, bd_factor, recenter
        )

        return img_paths, poses, hwf, min_bound, max_bound, path_poses

    @staticmethod
    def __normalize(v: np.ndarray) -> np.ndarray:
        """
        Normalizes a vector.
        ------------------------------------------------------------------------
        Args:
            v (np.array): [N,]. Vector
        Returns:
            v (np.array): [N,]. Normalized vector
        """
        return v / np.linalg.norm(v)

    @staticmethod
    def __viewmatrix(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
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
        z = LLFFDataset.__normalize(z)
        y = up
        x = LLFFDataset.__normalize(np.cross(y, z))
        y = LLFFDataset.__normalize(np.cross(z, x))
        matrix = np.stack([x, y, z, pos], axis=1)

        return matrix

    @staticmethod
    def __avg_pose(poses: np.ndarray) -> np.ndarray:
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
        viewdir = LLFFDataset.__normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate(
            [LLFFDataset.__viewmatrix(viewdir, up, center), hwf], 1
        )

        return c2w

    @staticmethod
    def __recenter_poses(poses: np.ndarray) -> np.ndarray:
        """
        Re-centers camera poses.
        ------------------------------------------------------------------------
        Args:
            poses (Tensor): [N, 3, 5]. Camera poses
        Returns:
            poses (Tensor): [N, 3, 5]. Re-centered camera poses
        """
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])  # last row of camera matrix
        c2w = LLFFDataset.__avg_pose(poses)  # average pose
        c2w = np.concatenate([c2w[:3, :4], bottom], axis=-2)  # center to world
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # camera to world

        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_

        return poses

    @classmethod
    def __postprocess_poses(
        cls,
        poses: np.ndarray,
        bounds: np.ndarray,
        bd_factor: float = 0.75,
        recenter: bool = True,
    ) -> Tuple[ndarray, Tuple[int, int, float], float, float, ndarray]:
        """
        Rescales and re-centers the scene poses, and derives the bounds,
        intrinsics and rendering spiral path from them.
        ------------------------------------------------------------------------
        Args:
            poses (ndarray): [N, 3, 5]. Camera poses with hwf in the last column
            bounds (ndarray): [N, 2]. Near and far bounds per view
            bd_factor (float): shrinks the scene so the nearest bound sits at
                               1 / bd_factor. None leaves the scale untouched
            recenter (bool): if True, expresses poses relative to the average pose
        Returns:
            poses (ndarray): [N, 3, 4]. Rescaled and re-centered camera poses
            hwf (Tuple[int, int, float]): image height, width and focal length
            min_bound (float): minimum value across the poses
            max_bound (float): maximum value across the poses
            path_poses (ndarray): [M, 3, 4]. Spiral path poses for rendering
                                  sample videos
        """
        # rescale bounds and poses
        scale = 1.0 if bd_factor is None else 1.0 / (bounds.min() * bd_factor)
        poses[..., :3, 3] *= scale
        bounds *= scale

        if recenter:
            poses = cls.__recenter_poses(poses)

        c2w = cls.__avg_pose(poses)
        path_poses = cls.__build_path(c2w, poses, bounds)  # for rendering video
        path_poses = np.stack(path_poses, 0)  # cast to numpy array
        path_poses = path_poses[:, :3, :4]

        hwf = poses[0, :3, -1]
        hwf = (int(hwf[0]), int(hwf[1]), float(hwf[2]))
        poses = poses[:, :3, :4]
        min_bound = poses.min()
        max_bound = poses.max()

        return poses, hwf, min_bound, max_bound, path_poses

    @staticmethod
    def __build_path(
        c2w: np.ndarray,
        poses: np.ndarray,
        bounds: np.ndarray,
        n_views: int = 60,
        n_rots: int = 2,
        zrate: float = 0.5,
        path_zflat: bool = False,
    ) -> Tensor:
        """
        Build spiral path for rendering sample video.
        ------------------------------------------------------------------------
        """
        up = LLFFDataset.__normalize(poses[:, :3, 1].sum(0))  # average up
        # compute reasonable focus depth for the scene
        close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # compute radii for spiral path
        shrink_factor = 0.8
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)

        if path_zflat:
            zloc = -close_depth * 0.1
            c2w[:3, 3] = c2w[:3, 3] + zloc * c2w[:3, 2]
            rads[2] = 0.0
            n_rots = 1
            n_views /= 2

        # compute spiral path
        path_poses = []
        rads = np.array(list(rads) + [1.0])
        hwf = c2w[:, 4:5]

        for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(
                c2w[:3, :4],
                np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
                * rads,
            )
            z = LLFFDataset.__normalize(
                c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
            )
            path_poses.append(
                np.concatenate([LLFFDataset.__viewmatrix(z, up, c), hwf], 1)
            )
        # cast to tensor
        return path_poses
