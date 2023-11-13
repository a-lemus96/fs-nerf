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
        raise NotImplementedError

class OccReg(Regularizer):
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

    def  __call__(self, sigmas: Tensor) -> Tensor:
        """
        Computes the mean density value within the regularization range multi-
        plied by the balancing weight.
        ------------------------------------------------------------------------
        Args:
            sigmas (Tensor): tensor of shape (..., K) denoting the density
                             values at the query points
        Returns:
            Tensor: regularization term multiplied by the balancing weight
        """
        # compute the loss
        reg = torch.mean(sigmas[..., :self.M], dim=-1)
        return self.weight * torch.mean(reg)
