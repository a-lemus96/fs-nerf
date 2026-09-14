from torch.optim import Optimizer


class LrScheduler:
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
        self.optim = optim
        if lro < 0:
            raise ValueError("lro must be a positive value.")
        self.lro = lro
        self.T = T
        self.t = 0  # current step
        self.r = r
        self.lrf = self.lro * self.r

    @property
    def lr(self) -> float:
        """Compute the learning rate."""
        lro, lrf = self.lro, self.lrf
        t, T = self.t, self.T

        return lro * (self.r ** (t / T)) if t < T else lrf

    def step(self) -> None:
        """
        Update optimizer learning rate
        ------------------------------------------------------------------------
        """
        self.t += 1
        for param_group in self.optim.param_groups:
            param_group["lr"] = self.lr
