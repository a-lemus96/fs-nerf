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


class VarianceRegularizer(OcclusionRegularizer):
    """
    Variance-based occlusion regularizer.

    Penalizes the variance of the rendering weight distribution along each
    ray, encouraging the model to place geometry at a single depth rather
    than spreading density across multiple samples. This biases the model
    toward thin, solid surface-like structures.

    For a ray with rendering weights {w_k} and midpoint depths {t_k}, the
    per-ray variance is:

        Var[t] = E[t^2] - E[t]^2
               = sum_k(w_k * t_k^2) - (sum_k(w_k * t_k))^2

    The loss is the mean variance across all rays in the batch:

        L_var = (1 / |R|) * sum_{r in R} Var_r[t]

    In NDC space, t values lie in [0, 1), so variance values are naturally
    small. The effect of this regularizer is stronger near the camera due
    to the nonlinear NDC compression of depth.
    """

    def __call__(self, result: RenderingResult) -> Tensor:
        """
        Computes the mean variance of the rendering weight distribution
        across all rays in the batch using a vectorized scatter-based approach.

        Args:
            result (RenderingResult): outputs from a volumetric rendering pass
        Returns:
            Tensor: scalar mean variance loss
        """
        weights     = result.weights
        t_vals      = result.t_vals
        ray_indices = result.ray_indices
        n_rays      = result.n_rays

        e_t = torch.zeros(n_rays, device=weights.device)
        e_t.scatter_add_(0, ray_indices, weights * t_vals)

        e_t2 = torch.zeros(n_rays, device=weights.device)
        e_t2.scatter_add_(0, ray_indices, weights * t_vals ** 2)

        var = (e_t2 - e_t ** 2).clamp(min=0.0)

        return var.mean()