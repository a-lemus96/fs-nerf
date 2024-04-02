# third-party related imports
import torch
from torch import Tensor
import torch.nn.functional as F

class OcclusionRegularizer():
    """
    Occlussion regularizer to penalize dense fields near the camera.
    ----------------------------------------------------------------------------
    """
    def __init__(self, a: float, b: float):
        """
        Initializes the occlusion regularizer.
        ------------------------------------------------------------------------
        Args:
            a (float): bias of the regularizer
            b (float): factor of the regularizer
        """
        self.a = a
        self.b = b
    
    def __call__(self, sigmas: Tensor, t_vals: Tensor) -> Tensor:
        """
        Computes occlussion regularization term for a batch of density values
        and their corresponding depths.
        ------------------------------------------------------------------------
        Args:
            sigmas (Tensor): density values of shape (B, N)
            t_vals (Tensor): depth values of shape (B, N)
        Returns:
            Tensor: occlusion regularization term of shape (B)
        """
        occlusion = (-a * t_vals + b) * sigmas
        return occlusion.sum(dim=-1)
