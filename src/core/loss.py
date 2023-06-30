# third-party related imports
import torch
from torch import Tensor
import torch.nn.functional as F


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

def depth_l1(
        depth: Tensor,
        depth_gt: Tensor
        ) -> Tensor:
    """Computes the depth loss between a batch of predicted and ground truth
    depth. Loss is defined by the sum of L1 norm between each pair of pixel
    depth. Background ground truth pixels are set to 0.
    ----------------------------------------------------------------------------
    Args:
        depth (Tensor): (B,). Predicted depth for a batch of rays
        depth_gt (Tensor): (B,). Ground truth depth for a batch of rays
    Returns:
        loss (Tensor): (1,). Depth loss for the batch of rays
    ----------------------------------------------------------------------------
    """
    bkgd = torch.isinf(depth_gt) # background pixels
    depth_gt[bkgd] = 0.0 # set gt background pixels to 0
    loss = F.l1_loss(depth, depth_gt) # compute L1 loss

    return loss
