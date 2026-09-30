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
    Exponential decay learning rate scheduler, as used in FreeNeRF.
    ----------------------------------------------------------------------------
    """

    def __init__(self, optim: Optimizer, T: int, lro: float, r: float) -> None:
        """
        Initialize the scheduler.
        ------------------------------------------------------------------------
        Args:
            optim (Optimizer): The optimizer to use
            T (int): The number of steps to decay the learning rate over
            lro (float): The initial learning rate
            r (float): Decay rate; the learning rate reaches lro * r at t = T
        Returns:
            None
        ------------------------------------------------------------------------
        """
        super().__init__(optim, lro)
        self.T = T
        self.r = r
        self.lrf = self.lro * self.r

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        lro, lrf = self.lro, self.lrf
        t, T = self.t, self.T

        return lro * (self.r ** (t / T)) if t < T else lrf
