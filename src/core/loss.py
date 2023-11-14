# third-party related imports
import torch
from torch import Tensor
import torch.nn.functional as F

class Regularizer:
    """
    Base class for regularizer.
    ----------------------------------------------------------------------------
    """
    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight (float): weight for the regularizer
        """
        self.weight = weight

    def __call__(self, *args, **kwargs):
        return self.weight * self.penalty(*args, **kwargs)

    @property
    def penalty(self, *args, **kwargs):
        raise NotImplementedError

class OcclusionRegularizer(Regularizer):
    """
    Occlussion regularizer to penalize dense fields near the camera.
    ----------------------------------------------------------------------------
    Reference(s):
        [1] Yang, J., Pavone, M., Wang, Y. "FreeNeRF: Improving Few-shot Neural 
        Rendering with Free Frequency Regularization." Proceedings of the 
        IEEE/CVF International Conference on Computer Vision (CVPR), 2023.
    """
    def __init__(self, weight: float = 1.0, M: int = 25):
        """
        Args:
            weight (float): weight for the regularizer
            M (int): index denoting the regularization range
        """
        super().__init__(weight)
        self.M = M

    @property
    def penalty(self, sigmas: Tensor, ray_indices: Tensor) -> Tensor:
        """
        Computes the mean density value within the regularization range for a
        batch of rays.
        ------------------------------------------------------------------------
        Args:
            sigmas (Tensor): (K,) denoting the density values at query points
        Returns:
            Tensor: (1,) mean density value within the regularization range
        """
        samples_per_ray = torch.bincount(ray_indices)
        nonzero_idxs = torch.nonzero(samples_per_ray).view(-1)
        splits = samples_per_ray[nonzero_idxs]
        s_groups = torch.split(sigmas, splits.tolist())
        sigmas = torch.cat([s[:min(self.M, len(s))] for s in s_groups])

        return self.weight * torch.mean(sigmas)
