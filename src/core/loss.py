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
        depth: Tensor,
        depth_gt: Tensor,
        weight: Tensor,
        z_val: Tensor
        ) -> Tensor:
    """Computes the depth loss between a batch of predicted and ground truth
    depth. Loss is defined by the sum of L1 norm between each pair of pixel
    depth divided by the square root of the variance as computed in [1].
    ----------------------------------------------------------------------------
    Reference(s):
        [1] Dey, A., Ahmine, Y., & Comport, A. I. (2022). Mip-NeRF RGB-D: Depth
        Assisted Fast Neural Radiance Fields. arXiv preprint arXiv:2205.09351.
    ----------------------------------------------------------------------------
    Args:
        depth (Tensor): (B,). Predicted depth for a batch of rays
        depth_gt (Tensor): (B,). Ground truth depth for a batch of rays
        weight (Tensor): (B, N). Weight distribution for each ray in the batch.
                          Here, N is the number of samples along each ray
        z_val (Tensor): (B, N). Depth values for each sample along each ray
    Returns:
        loss (Tensor): (1,). Depth loss for the batch of rays
    ----------------------------------------------------------------------------
    """
    # compute variance of depth distribution
    var = torch.sum(weight * (depth[..., None] - z_val)**2, dim=-1)
    bkgd = torch.isinf(depth_gt) # background pixels
    is_zero = torch.isclose(var, torch.zeros_like(var), atol=1e-6) # 0-var pixs
    idxs = ~is_zero
    depth_gt[bkgd] = 0.0 # set gt background pixels to 0
    # filter out 0-var pixels
    depth_gt = depth_gt[idxs]
    depth = depth[idxs]
    var = var[idxs]
    # compute depth loss
    loss = torch.mean(torch.abs(depth - depth_gt) / torch.sqrt(var))

    return loss
