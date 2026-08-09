import os
import numpy as np

import imageio as iio
from torch import Tensor

from ..datasets import llff

LLFF_HOLD = 8  # every LLFF_HOLD-th view is held out for testing


class Splitter:
    """
    Create train/test splits for a NeRF dataset following the FreeNeRF protocol.

    The split is deterministic and reproduces the one used by FreeNeRF/RegNeRF,
    so metrics are directly comparable to the numbers published for those methods:

    - Test split: every LLFF_HOLD-th view (indices 0, 8, 16, ...). It is evaluated
      in full and is currently reused as the validation split by the caller.
    - Train pool: the remaining views.
    - Few-shot train split: `n_views` evenly spaced picks over the train pool. LLFF
      captures sweep the scene sequentially, so evenly spaced indices translate to
      evenly spaced camera poses. Passing a negative `n_views` keeps the whole pool.

    Only the LLFF dataset layout is supported. Usage::

        splitter = Splitter("fern")
        splitter.split(n_views=3)
        train_dataset, test_dataset = splitter.get_datasets(ndc=True)
    """

    def __init__(self, scene: str):
        """
        Loads the scene and prepares its poses, bounds and intrinsics.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene folder name under ../datasets/llff/
        """
        self.scene = scene

        self.image_paths = []
        self.poses = np.empty((0, 3, 4))

        self._load_dataset()

    def split(self, n_views: int = -1):
        """
        Computes the train and test view indices for the scene.
        ------------------------------------------------------------------------
        Args:
            n_views (int): number of training views to keep. A negative value
                           keeps every view left in the train pool
        Returns:
            None. Sets self.train_ids and self.test_ids
        """
        all_indices = np.arange(len(self.poses))
        self.test_ids = all_indices[all_indices % LLFF_HOLD == 0]
        pool = all_indices[all_indices % LLFF_HOLD != 0]

        if n_views < 0:
            self.train_ids = pool
        else:
            assert 0 < n_views <= len(pool), (
                f"ValueError, the number of training views must be in [1, {len(pool)}], "
                f"got {n_views}."
            )
            idx_sub = [round(i) for i in np.linspace(0, len(pool) - 1, n_views)]
            self.train_ids = pool[idx_sub]

    def get_datasets(self, train_img_mode: bool = False, **kwargs):
        """
        Builds the train and test datasets from a prior call to split().
        ------------------------------------------------------------------------
        Args:
            train_img_mode (bool): if True, the training dataset yields whole
                                   images instead of rays. The test dataset is
                                   always built in image mode
            **kwargs: forwarded dataset options. 'ndc' (bool, default False)
                      maps rays to normalized device coordinates
        Returns:
            train_dataset (LLFFDataset): dataset over the training views
            test_dataset (LLFFDataset): dataset over the held-out views
        """
        assert (
            self.train_ids is not None
        ), "Split the source data before building the datasets."
        # Get image files
        test_poses = self.poses[self.test_ids]
        test_img_paths = self.img_paths[self.test_ids]
        test_imgs = self._load_img_files(test_img_paths)

        train_poses = self.poses[self.train_ids]
        train_img_paths = self.img_paths[self.train_ids]
        train_imgs = self._load_img_files(train_img_paths)

        # Instantiate datasets
        ndc = kwargs.get("ndc", False)
        test_dataset = llff.LLFFDataset(
            test_imgs,
            test_poses,
            self.min_bound,
            self.max_bound,
            self.hwf,
            True,
            ndc,
        )

        train_dataset = llff.LLFFDataset(
            train_imgs,
            train_poses,
            self.min_bound,
            self.max_bound,
            self.hwf,
            train_img_mode,
            ndc,
        )

        return train_dataset, test_dataset

    def _load_dataset(self):
        """
        Load image paths, camera poses, bounds and intrinsics from an llff
        dataset folder.
        ------------------------------------------------------------------------
        Expected scene folder structure:
            images_8/          -> contains image files
            poses_bounds.npy   -> poses file
        """
        base_folder_path = os.path.normpath("../datasets/llff/")
        assert os.path.isdir(
            base_folder_path
        ), f"LLFF dataset folder {os.path.abspath(base_folder_path)} not found."
        all_scenes = os.listdir(path=base_folder_path)
        assert (
            self.scene in all_scenes
        ), f"Scene '{self.scene}' not found in local LLFF dataset folder."

        # load camera poses and bounds
        base_scene_folder_path = os.path.join(base_folder_path, self.scene)
        data = np.load(os.path.join(base_scene_folder_path, "poses_bounds.npy"))
        poses = data[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bounds = data[:, -2:].transpose([1, 0])

        # load the downsampled images
        imgs_folder_path = os.path.normpath(
            os.path.join(base_folder_path, self.scene, "images_8/")
        )
        assert os.path.isdir(
            imgs_folder_path
        ), f"Images folder path {os.path.abspath(imgs_folder_path)} not found."
        img_paths = [
            os.path.abspath(os.path.join(imgs_folder_path, f))
            for f in sorted(os.listdir(imgs_folder_path))
            if f.endswith(("JPG", "jpg", "png"))
        ]
        self.img_paths = np.array(img_paths)
        assert (
            len(self.img_paths) == poses.shape[-1]
        ), "Mismath between the number of images and poses"

        # modify camera poses
        H, W, _ = iio.imread(self.img_paths[0]).shape
        poses[:2, 4, :] = np.array([H, W]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / 8.0
        # correct poses ordering
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], axis=1
        )

        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)

        self.postprocess_poses(poses, bounds)

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
        z = Splitter.__normalize(z)
        y = up
        x = Splitter.__normalize(np.cross(y, z))
        y = Splitter.__normalize(np.cross(z, x))
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
        viewdir = Splitter.__normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([Splitter.__viewmatrix(viewdir, up, center), hwf], 1)

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
        c2w = Splitter.__avg_pose(poses)  # average pose
        c2w = np.concatenate([c2w[:3, :4], bottom], axis=-2)  # center to world
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # camera to world

        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_

        return poses

    def postprocess_poses(
        self,
        poses: np.ndarray,
        bounds: np.ndarray,
        factor: int = 4,
        bd_factor: float = 0.75,
        recenter: bool = True,
        ndc: bool = True,
    ):
        """
        Rescales, re-centers and stores the scene poses, bounds and intrinsics.
        ------------------------------------------------------------------------
        Args:
            poses (ndarray): [N, 3, 5]. Camera poses with hwf in the last column
            bounds (ndarray): [N, 2]. Near and far bounds per view
            bd_factor (float): shrinks the scene so the nearest bound sits at
                               1 / bd_factor. None leaves the scale untouched
            recenter (bool): if True, expresses poses relative to the average pose
        Returns:
            None. Sets self.poses, self.path_poses, self.hwf, self.min_bound and
            self.max_bound
        """
        # rescale bounds and poses
        scale = 1.0 if bd_factor is None else 1.0 / (bounds.min() * bd_factor)
        poses[..., :3, 3] *= scale
        bounds *= scale

        if recenter:
            poses = Splitter.__recenter_poses(poses)

        c2w = Splitter.__avg_pose(poses)
        path_poses = self.__build_path(c2w, poses, bounds)  # for rendering video
        path_poses = np.stack(path_poses, 0)  # cast to numpy array
        self.path_poses = path_poses[:, :3, :4]

        hwf = poses[0, :3, -1]
        self.hwf = (int(hwf[0]), int(hwf[1]), float(hwf[2]))
        self.poses = poses[:, :3, :4]
        self.min_bound = poses.min()
        self.max_bound = poses.max()

    def _load_img_files(self, img_paths):
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

    def __build_path(
        self,
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
        up = Splitter.__normalize(poses[:, :3, 1].sum(0))  # average up
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
            z = Splitter.__normalize(
                c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
            )
            path_poses.append(np.concatenate([Splitter.__viewmatrix(z, up, c), hwf], 1))
        # cast to tensor
        return path_poses
