# third-party related imports
import torch

# REGULARIZATION TERMS

def gini_entropy(
        sigmas: torch.Tensor
        ) -> torch.Tensor:
    r"""Regularization function based on Gini's impurity index. Specifically,
    the negative of the Herfindahl index for the proportions of density along a
    batch of rays.
    ----------------------------------------------------------------------------
    Args:
        sigmas: (B, N)-shape tensor. Contains a batch of size B rays each with N
                density values
    Returns:
        loss: (1,)-shape tensor. Contains negative of Herfindahl index for the
        supplied data"""
    # compute density proportions for each ray
    total = torch.sum(sigmas, dim=-1)
    sigmas = sigmas / total[..., None]
    
    # compute sum of squared proportions
    loss = -torch.sum(sigmas**2, dim=-1)

    return loss 
