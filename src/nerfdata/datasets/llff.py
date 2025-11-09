# standard library modules
from typing import Tuple
from dataclasses import dataclass

# third-party modules
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

    def __init__(
        self,
        imgs: np.array,
        poses: np.array,
        min_bound: float,
        max_bound: float,
        hwf: Tuple[int, int, float],
        white_bkgd: bool = False,
        img_mode: bool = False,
        ndc: bool = True,
    ) -> None:
        """
        Initializes the dataset.
        ------------------------------------------------------------------------
        Args:
            To be updated...
        """
        super(LLFFDataset, self).__init__()
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.poses = torch.tensor(poses, dtype=torch.float32)
        self.hwf = hwf
        self.white_bkgd = white_bkgd
        self.img_mode = img_mode
        self.ndc = ndc

        # Define the ray bounds
        if not ndc:
            self.near = min_bound * 0.9
            self.far = max_bound * 1.0
        else:
            self.near = 0.0
            self.far = 1.0

        # Build ray-rgb g.t. samples
        if not self.img_mode:
            self.__build_samples()

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
