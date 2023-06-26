# third-party related imports
import torch
from torch import Tensor

# REGULARIZATION TERMS

def gini_entropy(
        sigmas: Tensor
        ) -> Tensor:
    r"""Regularization function based on Gini's impurity index. Specifically,
    the negative of the Herfindahl index for the proportions of density along a
    batch of rays.
    ----------------------------------------------------------------------------
    Args:
        sigmas (Tensor): (B, N). Density values fora batch of rays. Here, N is
                         the number of samples along each ray
    Returns:
        loss (Tensor): (1,). Gini entropy loss for the batch of rays
    ----------------------------------------------------------------------------
    """
    # compute density proportions for each ray
    total = torch.sum(sigmas, dim=-1)
    sigmas = sigmas / total[..., None]
    
    # compute sum of squared proportions
    loss = torch.mean(-torch.sum(sigmas**2, dim=-1))

    return loss 

def depth(
        depths: Tensor,
        depths_gt: Tensor,
        weights: Tensor
        ) -> Tensor:
    """Computes the depth loss between a batch of predicted and ground truth
    depths. Loss is defined by the sum of L1 norm between each pair of pixel
    depths divided by the square root of the variance as computed in [1].
    ----------------------------------------------------------------------------
    Reference(s):
        [1] Dey, A., Ahmine, Y., & Comport, A. I. (2022). Mip-NeRF RGB-D: Depth
        Assisted Fast Neural Radiance Fields. arXiv preprint arXiv:2205.09351.
    ----------------------------------------------------------------------------
    Args:
        depths (Tensor): (B,). Predicted depths for a batch of rays
        depths_gt (Tensor): (B,). Ground truth depths for a batch of rays
        weights (Tensor): (B, N). Weights distribution for each ray in the batch.
                          Here, N is the number of samples along each ray.
    Returns:
        loss (Tensor): (1,). Depth loss for the batch of rays
    ----------------------------------------------------------------------------
    """
    # compute variance of depth distribution
    var = torch.sum(weights * (depths - depths_gt)**2, dim=-1)
    loss = torch.mean(torch.abs(depths - depths_gt) / torch.sqrt(var))

    return loss
