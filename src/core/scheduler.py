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
            lr_range: tuple = (5e-4, 5e-5)
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            steps (int): The number of steps to decay the learning rate over
            lr_range (tuple): The range of learning rates to decay between
        Returns:
            None
        ------------------------------------------------------------------------
        """
        self.optim = optim
        self.lr_max, self.lr_min = lr_range
        if self.lr_max <= self.lr_min:
            raise ValueError("lr_max must be greater than lr_min.")
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

class ExponentialDecay(Scheduler):
    """
    Exponential decay for the learning rate
    ----------------------------------------------------------------------------
    """
    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        decay_rate = -np.log(self.lr_min / self.lr_max) / self.steps
        return self.lr_max * np.exp(-decay_rate * self.current_step)

class RootP(Scheduler):
    """p-root-based learning rate decay."""
    def __init__(
            self,
            optim: Optimizer,
            steps: int,
            lr_range: tuple = (5e-4, 5e-5),
            p: int = 2
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            steps (int): The number of steps to decay the learning rate over
            lr_range (tuple): The range of learning rates to iterate between
            p (int): The p-root to use
        Returns:
            None
        ------------------------------------------------------------------------
        """
        super().__init__(optim, steps, lr_range)
        if p == 0:
            raise ValueError("p must be different from 0.")
        self.p = int(p)

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        p, N, k = self.p, self.steps, self.current_step
        if k < N:
            t = (((1. - 0.5 ** p) / N) * k + 0.5 ** p) ** (1. / p)
            lr = 2 * (self.lr_max - self.lr_min) * (1. - min(1., t))
            lr += self.lr_min
        else:
            lr = self.lr_min
        
        return lr

class MipNerf(Scheduler):
    """MipNerf learning rate scheduler."""
    def __init__(
            self,
            optim: Optimizer,
            steps: int,
            warmup_steps: int = 2500,
            lr_range: tuple = (5e-4, 5e-6),
            scale: float = 0.01
    ) -> None:
        """Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            steps (int): The number of steps to decay the learning rate over
            warmup_steps (int): The number of steps to warmup the learning rate
            lr_range (tuple): The range of learning rates to decay logarithmica-
                              lly between.
            scale (float): The scale factor to apply during warmup phase
        Returns:
            None
        ------------------------------------------------------------------------
        """
        super().__init__(optim, steps, lr_range)
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.scale = scale

    @property
    def lr(self) -> float:
        """Calculate the learning rate."""
        t = np.clip(self.current_step / self.warmup_steps, 0, 1)
        t = (1 - self.scale) * np.sin(0.5 * np.pi * t)
        factor = self.scale + t

        t = self.current_step / self.steps
        lr = np.exp((1 - t) * np.log(self.lr_max) + t * np.log(self.lr_min))
        return factor * lr
