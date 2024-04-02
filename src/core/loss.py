# third-party related imports
import torch
from torch import Tensor
import torch.nn.functional as F

class OcclusionRegularizer():
    """
    Occlussion regularizer to penalize dense fields near the camera.
    ----------------------------------------------------------------------------
    """
    def __init__(self, a: float, b: float, type: str = 'linear'):
        """
        Initializes the occlusion regularizer.
        ------------------------------------------------------------------------
        Args:
            a (float): bias of the regularizer
            b (float): factor of the regularizer
            type (str): type of the occlusion regularizer
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
        occlusion = self._weights(t_vals) * sigmas
        return occlusion.sum(dim=-1)

    def _weights(self, t_vals: Tensor) -> Tensor:
        """
        Computes importance weights for occlusion regularization.
        ------------------------------------------------------------------------
        Args:
            t_vals (Tensor): depth values of shape (B, N)
        Returns:
            Tensor: importance weights of shape (B, N)
        """
        match self.type:
            case 'linear':
                weights = -self.a * t_vals + self.b
            case 'exp':
                weights = self.a * torch.exp(-self.b * t_vals)
            case _:
                raise ValueError(f'Unknown occlusion regularizer type: {self.type}')
        return weights
