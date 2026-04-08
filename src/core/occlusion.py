# standard library imports
from abc import ABC, abstractmethod

# third-party imports
import torch
from torch import Tensor

from render.rendering import RenderingResult

class OcclusionRegularizer(ABC):
    """
    Abstract base class for occlusion regularizers.

    Occlusion regularizers penalize undesirable density distributions along
    rays, encouraging the model to converge toward geometrically well-behaved
    solutions in the few-shot setting.

    Subclasses must implement __call__, which accepts a RenderingResult and
    returns a scalar regularization loss.
    """

    @abstractmethod
    def __call__(self, result: RenderingResult) -> Tensor:
        """
        Computes a scalar regularization loss from a RenderingResult.

        Args:
            result (RenderingResult): outputs from a volumetric rendering pass
        Returns:
            Tensor: scalar regularization loss
        """
        pass

    
class WeightSumSquaredRegularizer(OcclusionRegularizer):
    """
    Weight-sum-squared occlusion regularizer.

    Encourages each ray to concentrate its rendering weight on as few samples
    as possible, biasing the model toward opaque, surface-like geometry rather
    than translucent or foggy density distributions.

    For a ray r with per-sample rendering weights {w_k}, the per-ray score is:

        S_r = sum_k w_{r,k}^2

    This is maximized when all weight is on a single sample (S_r = 1) and
    minimized when weight is spread uniformly across N samples (S_r = 1/N).

    The loss is the *negative* mean score across all rays in the batch, so
    that minimizing the loss encourages peaked (concentrated) distributions:

        L_occ = -(1 / |R|) * sum_{r in R} S_r

    Implementation uses scatter_add_ for efficiency over the packed-ray format
    produced by nerfacc.
    """

    def __call__(self, result: RenderingResult) -> Tensor:
        """
        Computes the negative mean per-ray sum-of-squared rendering weights.

        Args:
            result (RenderingResult): outputs from a volumetric rendering pass
        Returns:
            Tensor: scalar regularization loss (negative, to be minimized)
        """
        weights     = result.weights       # (S,)
        ray_indices = result.ray_indices   # (S,)
        n_rays      = result.n_rays

        # Accumulate sum of squared weights per ray: shape (n_rays,)
        sum_sq = torch.zeros(n_rays, device=weights.device)
        sum_sq.scatter_add_(0, ray_indices, weights ** 2)

        # Negate so that minimizing loss maximizes weight concentration
        return -sum_sq.mean()