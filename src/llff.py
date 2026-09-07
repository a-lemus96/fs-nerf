# standard library modules
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
from utils import load_scene


class LLFFDataset(Dataset):
    """
    Represents an instance of a Local Light Field Fusion dataset.
    ----------------------------------------------------------------------------
    """

    def __init__(
        self,
        scene: str,
        batch_size: int = 1024,
        img_ids: list[int] = None,
    ) -> None:
        """
        Loads the scene: images, poses, bounds and intrinsics. Uses NDC to map
        rays to normalized device coordinates.
        ------------------------------------------------------------------------
        Args:
            scene (str): scene folder name under ../datasets/llff/
            batch_size (int): number of rays sampled per image in __getitem__
            img_ids (list[int]): if given, restricts the dataset to these
                image indices (computed over the full, unfiltered scene) after
                the full scene has been loaded and its poses normalized
        """
        super(LLFFDataset, self).__init__()
        (
            img_paths,
            poses,
            self.hwf,
            self.min_bound,
            self.max_bound
        ) = load_scene(scene)

        if img_ids is not None:
            img_paths = img_paths[img_ids]
            poses = poses[img_ids]

        # set ray bounds for NDC
        self.near = 0.0
        self.far = 1.0
        self.batch_size = batch_size

        # build rays and get aabb
        images = torch.tensor(LLFFDataset.load_img_files(img_paths), dtype=torch.float32)
        poses = torch.tensor(poses, dtype=torch.float32)
        self.rays_d, self.rays_o, self.aabb = self.__build_samples(images, poses)

    def __build_samples(self, images, poses) -> tuple[Tensor, Tensor, Tensor]:
        """
        Builds rays and samples.
        ------------------------------------------------------------------------
        """
        H, W, _ = self.hwf
        N = len(images)
        self.rgb = images.reshape(N, -1, 3)  # [N, H * W, 3]

        rays = torch.stack(
            [torch.cat(U.get_rays(p, self.hwf), -1) for p in poses], 0
        )
        rays = rays.reshape(N, -1, 6)  # [N, H * W, 6]
        rays_o = rays[..., :3]  # ray origins, [N, H * W, 3]
        rays_d = rays[..., 3:]  # ray directions, [N, H * W, 3]

        # map to ndc
        rays_o, rays_d = U.to_ndc(rays_o, rays_d, self.hwf, 1.0)
        flat_o, flat_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        min_roi = torch.vstack(
            [flat_o.min(dim=0)[0], (flat_o + flat_d).min(dim=0)[0]]
        ).min(dim=0)[0]
        max_roi = torch.vstack(
            [flat_o.max(dim=0)[0], (flat_o + flat_d).max(dim=0)[0]]
        ).max(dim=0)[0]
        aabb = torch.hstack([min_roi, max_roi])
        aabb = aabb / 2 ** (4 - 1)

        return rays_o, rays_d, aabb

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
        self.rays_o = self.rays_o.to(device)
        self.rays_d = self.rays_d.to(device)
        self.rgb = self.rgb.to(device)
        
        return self

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Samples a batch of rays from the idx-th image.
        ------------------------------------------------------------------------
        Args:
            idx (int): index of the image to sample from
        Returns:
            ray_o (Tensor): [batch_size, 3]. Ray origins
            ray_d (Tensor): [batch_size, 3]. Ray directions
            rgb (Tensor): [batch_size, 3]. Pixel RGB colors
        """
        n_pixels = self.rays_o.shape[1]
        pixel_idxs = torch.randint(
            0, n_pixels, (self.batch_size,), device=self.rays_o.device
        )

        return (
            self.rays_o[idx, pixel_idxs],
            self.rays_d[idx, pixel_idxs],
            self.rgb[idx, pixel_idxs],
        )

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
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
