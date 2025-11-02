import os
import numpy as np
from typing import Literal, List, Tuple, Optional

import imageio as iio
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans

from ..datasets import llff


class Splitter:
    """
    Create train/validation/test splits for a NeRF dataset based on cam poses.

    Supports two strategies for creating the training split:
    - 'random': randomly samples the images for the training split.
    - 'pose_based': selects diverse subsets using pose distances.

    Note: The split strategy applies to the training split only. Validation and
    test splits are sampled using the 'pose_based' strategy.
    """

    def __init__(
        self,
        dataset_type: str,
        scene: str,
        strategy: str = "pose_based",
        n_training_views=-1,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = None,
    ):
        self.dataset_type = dataset_type
        self.scene = scene
        self.strategy = strategy
        self.n_training_views = n_training_views
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.image_paths = []
        self.poses = np.empty((0, 3, 4))

        self._load_dataset()

    def split(self):
        available_idxs = np.arange(len(self.poses))
        n_test_samples = int(self.test_ratio * len(self.poses))
        self.test_ids, available_idxs = self._select_pose_based(
            available_idxs, n_test_samples)
    
        n_val_samples = int(self.val_ratio * len(self.poses))
        self.val_ids, available_idxs = self._select_pose_based(
            available_idxs, n_val_samples
        )
        if self.n_training_views < 0:
            self.train_ids = available_idxs
        else:
            assert (
                self.n_training_views > 0
            ), "ValueError, the specified number of training images must be greater than zero."
            self.train_ids, _ = self._select_pose_based(
                available_idxs, self.n_training_views
            )

    def get_datasets(self, train_img_mode: bool = False, **kwargs):
        """Instantiates and returns datasets based on a prior split. Img mode works for training only."""
        assert (
            self.train_ids is not None
        ), "Split the source data before building the datasets."
        # Get image files
        test_poses = self.poses[self.test_ids]
        test_img_paths = self.img_paths[self.test_ids]
        test_imgs = self._load_files(test_img_paths)

        val_poses = self.poses[self.val_ids]
        val_img_paths = self.img_paths[self.val_ids]
        val_imgs = self._load_files(val_img_paths)

        train_poses = self.poses[self.train_ids]
        train_img_paths = self.img_paths[self.train_ids]
        train_imgs = self._load_files(train_img_paths)

        # Instantiate datasets
        white_bkgd = kwargs.get("white_bkgd", False)
        ndc = kwargs.get("ndc", True)
        test_dataset = llff.LLFFDataset(
            test_imgs, test_poses, self.hwf, white_bkgd, True, ndc
        )
        val_dataset = llff.LLFFDataset(
            val_imgs, val_poses, self.hwf, white_bkgd, True, ndc
        )
        train_dataset = llff.LLFFDataset(
            train_imgs, train_poses, self.hwf, white_bkgd, train_img_mode, ndc
        )

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(self, train_batch_size, train_img_mode=False, **kwargs):
        datasets = self.get_datasets(train_img_mode, **kwargs)
        train_set, val_set, test_set = datasets
        train_loader = DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True, num_workers=8
        )
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8)

        return train_loader, val_loader, test_loader

    def _select_pose_based(self, available_idxs: np.ndarray, n_samples: int):
        # apply K-means to position vectors
        x = self.poses[available_idxs, :3, 3]
        kmeans = KMeans(n_clusters=n_samples, n_init=10).fit(x)  # kmeans model
        labels = kmeans.labels_

        # compute distances of each sample to its cluster center
        dists = np.linalg.norm(x - kmeans.cluster_centers_[labels], axis=1)
        # choose the closest view for every cluster center
        idxs = np.empty((n_samples,), dtype=int)  # array for indices of views
        for i in range(n_samples):
            cluster_dists = np.where(labels == i, dists, np.inf)
            idxs[i] = np.argmin(cluster_dists)

        selected_sample_idxs = available_idxs[idxs]
        #  remove the selected idxs
        new_available_idxs = []
        for i in range(len(available_idxs)):
            current_idx = available_idxs[i]
            if current_idx not in selected_sample_idxs:
                new_available_idxs.append(current_idx)

        return selected_sample_idxs, np.array(new_available_idxs)

    def _select_random_based(self):
        pass

    def _load_dataset(self):
        """
        Load image paths and corresponding camera poses from the dataset folder.
        The expected folder structure depends on the dataset type.
        """
        if self.dataset_type == "llff":
            self._load_llff_dataset()
        else:
            raise ValueError(f"Dataset of type '{self.dataset_type}' is not supported.")

    def _load_synth_dataset(self):
        pass

    def _load_llff_dataset(self):
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
        # rescale bounds and poses
        scale = 1.0 if bd_factor is None else 1.0 / (bounds.min() * bd_factor)
        poses[..., :3, 3] *= scale
        bounds *= scale

        if recenter:
            poses = Splitter.__recenter_poses(poses)

        # c2w = DatasetSplitter.__avg_pose(poses)
        # path_poses = self.__build_path(c2w, poses, bounds) # for rendering video
        # path_poses = np.stack(path_poses, 0) # cast to numpy array
        # path_poses = path_poses[:, :3, :4]

        hwf = poses[0, :3, -1]
        self.hwf = (int(hwf[0]), int(hwf[1]), float(hwf[2]))
        self.poses = poses[:, :3, :4]
        self.min_bound = poses.min()
        self.max_bound = poses.max()

    def _load_files(self, img_paths):
        pass

    def _validate_ratios(self):
        pass
