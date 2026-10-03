import math
from abc import ABC, abstractmethod

from torch.optim import Optimizer


class LrScheduler(ABC):
    """
    Abstract learning rate scheduler.
    ----------------------------------------------------------------------------
    """

    def __init__(self, optim: Optimizer, lro: float) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            lro (float): The initial learning rate
        Returns:
            None
        ------------------------------------------------------------------------
        """
        if lro < 0:
            raise ValueError("lro must be a positive value.")
        self.optim = optim
        self.lro = lro
        self.t = 0  # current step

    @property
    @abstractmethod
    def lr(self) -> float:
        """Learning rate at the current step."""
        pass

    def step(self) -> None:
        """
        Advance one step and update optimizer learning rate
        ------------------------------------------------------------------------
        """
        self.t += 1
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr


class ConstantLrScheduler(LrScheduler):
    """
    Constant learning rate scheduler: outputs the initial learning rate at
    every step.
    ----------------------------------------------------------------------------
    """

    @property
    def lr(self) -> float:
        """Return the initial learning rate."""
        return self.lro


class ExponentialLrScheduler(LrScheduler):
    """
    Exponential decay learning rate scheduler, as used in FreeNeRF. An optional
    warmup multiplier scales the decay curve: it rises from warmup_mult to 1
    along a quarter sine wave over warmup_iters steps, then stays at 1.
    ----------------------------------------------------------------------------
    """

    def __init__(
        self,
        optim: Optimizer,
        T: int,
        lro: float,
        r: float,
        warmup_iters: int = 0,
        warmup_mult: float = 1.0,
    ) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            T (int): The number of steps to decay the learning rate over
            lro (float): The initial learning rate
            r (float): Decay rate; the learning rate reaches lro * r at t = T
            warmup_iters (int): Steps over which the warmup multiplier reaches
                1; 0 disables warmup
            warmup_mult (float): Multiplier on the learning rate at t = 0
        Returns:
            None
        ------------------------------------------------------------------------
        """
        super().__init__(optim, lro)
        self.T = T
        self.r = r
        self.lrf = self.lro * self.r
        self.warmup_iters = warmup_iters
        self.warmup_mult = warmup_mult
        # Apply the t = 0 rate so the first optimizer step is already warmed up
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr

    @property
    def warmup_factor(self) -> float:
        """Warmup multiplier at the current step."""
        if self.warmup_iters <= 0 or self.t >= self.warmup_iters:
            return 1.0
        x = self.t / self.warmup_iters
        return self.warmup_mult + (1 - self.warmup_mult) * math.sin(0.5 * math.pi * x)

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        lro, lrf = self.lro, self.lrf
        t, T = self.t, self.T

        decay = lro * (self.r ** (t / T)) if t < T else lrf
        return self.warmup_factor * decay
