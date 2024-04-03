# third-party related imports
import torch
from torch import Tensor
import torch.nn.functional as F

class OcclusionRegularizer():
    """
    Occlussion regularizer to penalize dense fields near the camera.
    ----------------------------------------------------------------------------
    """
    def __init__(self, a: float, b: float, func: str = 'linear'):
        """
        Initializes the occlusion regularizer.
        ------------------------------------------------------------------------
        Args:
            a (float): bias of the regularizer
            b (float): factor of the regularizer
            func (str): type of the occlusion regularizer
        """
        assert a >= 0, 'a should be non-negative'
        self.a = a
        assert b >= 0, 'b should be non-negative'
        self.b = b
        self.func = func
    
    def __call__(self, sigmas: Tensor, 
                 t_vals: Tensor, ray_idxs: Tensor) -> Tensor:
        """
        Computes occlussion regularization term for a batch of density values
        and their corresponding depths.
        ------------------------------------------------------------------------
        Args:
            sigmas (Tensor): density values of shape (N,)
            t_vals (Tensor): depth values of shape (N,)
            ray_idxs (Tensor): ray indices of shape (N,)
        Returns:
            Tensor: occlusion regularization term of shape (1,)
        """
        uniques = torch.unique_consecutive(ray_idxs)
        occl = [torch.sum(self._weights(t_vals[ray_idxs == val]) * sigmas[ray_idxs == val])
                for val in uniques]
        return torch.mean(torch.stack(occl))

    def _weights(self, t_vals: Tensor) -> Tensor:
        """
        Computes importance weights for occlusion regularization.
        ------------------------------------------------------------------------
        Args:
            t_vals (Tensor): depth values of shape (B, N)
        Returns:
            Tensor: importance weights of shape (B, N)
        """
        match self.func:
            case 'linear':
                weights = -self.a * t_vals + self.b
            case 'exp':
                weights = self.a * torch.exp(-self.b * t_vals)
            case _:
                raise ValueError(f'Unknown occlusion regularizer type: {self.type}')
        return weights
