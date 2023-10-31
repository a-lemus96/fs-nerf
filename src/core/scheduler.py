# third-party imports
import numpy as np
import torch
from torch.optim import Optimizer

class Scheduler:
    """Learning rate scheduler."""
    def __init__(
            self,
            optim: Optimizer,
            steps: int,
            lr_range: tuple = (5e-4, 5e-5),
            **kwargs: dict
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            steps (int): The number of steps to decay the learning rate over
            lr_range (tuple): The range of learning rates to decay between
            **kwargs (dict): Additional keyword arguments
        Returns:
            None
        ------------------------------------------------------------------------
        """
        self.optim = optim
        self.lr_max, self.lr_min = lr_range
        if self.lr_max < self.lr_min:
            raise ValueError("lr_max must be greater than or equal to lr_min.")
        self.steps = steps
        self.current_step = 0

    def step(self) -> None:
        """
        Update optimizer learning rate
        ------------------------------------------------------------------------
        """
        self.current_step += 1
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
        return self.lr_max


class ExponentialDecay(Scheduler):
    """
    Exponential decay for the learning rate
    ----------------------------------------------------------------------------
    """
    def __init__(
            self,
            optim: Optimizer,
            steps: int,
            lr_range: tuple = (5e-4, 5e-5),
            **kwargs: dict
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        """
        super().__init__(optim, steps, lr_range)
        self.r = kwargs["r"]

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        #decay_rate = -np.log(self.lr_min / self.lr_max) / self.steps
        #return self.lr_max * np.exp(-decay_rate * self.current_step)
        return (self.r ** self.current_step) * self.lr_max


class RootP(Scheduler):
    """p-root-based learning rate decay."""
    def __init__(
            self,
            optim: Optimizer,
            steps: int,
            lr_range: tuple = (5e-4, 5e-5),
            **kwargs: dict
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            steps (int): The number of steps to decay the learning rate over
            lr_range (tuple): The range of learning rates to iterate between
            **kwargs (dict): Additional keyword arguments
        Returns:
            None
        ------------------------------------------------------------------------
        """
        super().__init__(optim, steps, lr_range)
        p = kwargs["p"]
        self.T_lr = kwargs["T_lr"]
        if int(p) == 0:
            raise ValueError("p must be different from 0.")
        self.p = int(p)

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        p, N, k = self.p, self.T_lr, self.current_step
        if k < N:
            t = (((1. - 0.5 ** p) / N) * k + 0.5 ** p) ** (1. / p)
            lr = 2 * (self.lr_max - self.lr_min) * (1. - min(1., t))
            lr += self.lr_min
        else:
            lr = self.lr_min
        
        return lr
