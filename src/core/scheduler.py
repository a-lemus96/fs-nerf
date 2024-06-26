# third-party imports
import numpy as np
import torch
from torch.optim import Optimizer

class Scheduler:
    """Learning rate scheduler."""
    def __init__(
            self,
            optim: Optimizer,
            T: int,
            lro: float,
            **kwargs: dict
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            T (int): The number of steps to decay the learning rate over
            lro (float): The initial learning rate
            **kwargs (dict): Additional keyword arguments
        Returns:
            None
        ------------------------------------------------------------------------
        """
        self.optim = optim
        if lro < 0:
            raise ValueError("lro must be a positive value.")
        self.lro = lro
        self.T = T
        self.t = 0 # current step

    def step(self) -> None:
        """
        Update optimizer learning rate
        ------------------------------------------------------------------------
        """
        self.t += 1
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr

class Constant(Scheduler):
    """
    Constant learning rate
    ----------------------------------------------------------------------------
    """
    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        return self.lro


class ExponentialDecay(Scheduler):
    """
    Exponential decay for the learning rate
    ----------------------------------------------------------------------------
    """
    def __init__(
            self,
            optim: Optimizer,
            T: int,
            lro: float,
            **kwargs: dict
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        """
        super().__init__(optim, T, lro)
        self.r = kwargs["r"]
        self.lrf = self.lro * self.r

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        lro, lrf = self.lro, self.lrf
        t, T = self.t, self.T

        return lro * (self.r ** (t / T)) if t < T else lrf
