# stdlib modules
from abc import ABC, abstractmethod

# third-party modules
import torch

# local modules
from .models import Sinerf


class FrequencySchedule:
    """
    Interpolates a frequency value from start to end over T training steps,
    then holds end. Interpolation is either linear or geometric ("log"); the
    latter mirrors FreeNeRF, which opens log-spaced frequency bands linearly.
    ----------------------------------------------------------------------------
    """

    def __init__(self, start: float, end: float, T: int, mode: str = "log") -> None:
        """
        Constructor method.
        ------------------------------------------------------------------------
        Args:
            start: float. Value at step 0
            end: float. Value at step T and after
            T: int. Number of steps over which to interpolate
            mode: str. Interpolation mode, one of 'linear' or 'log'
        Raises:
            ValueError: if mode is unknown, T < 1, or mode is 'log' with a
                non-positive start or end
        """
        if mode not in ("linear", "log"):
            raise ValueError(f"mode must be 'linear' or 'log', got '{mode}'")
        if T < 1:
            raise ValueError(f"T must be at least 1, got {T}")
        if mode == "log" and (start <= 0 or end <= 0):
            raise ValueError("'log' mode needs positive start and end values")
        self.start = start
        self.end = end
        self.T = T
        self.mode = mode

    def __call__(self, t: int) -> float:
        """
        Returns the scheduled value at step t.
        ------------------------------------------------------------------------
        Args:
            t: int. Current training step
        Returns:
            float. Scheduled value
        """
        x = min(t / self.T, 1.0)
        if self.mode == "linear":
            return self.start + (self.end - self.start) * x
        return self.start * (self.end / self.start) ** x


class FrequencyCurriculum(ABC):
    """
    Abstract frequency curriculum for SIREN-based NeRF models.

    Restricts the input frequencies of the first SIREN layer, starting from
    low frequencies and progressively allowing higher ones. Unlike a loss
    term, a curriculum acts directly on the model: it is applied once at
    construction and again after every optimizer step via step().
    ----------------------------------------------------------------------------
    """

    def __init__(self, model: Sinerf, schedule: FrequencySchedule) -> None:
        """
        Constructor method. Applies the curriculum at step 0.
        ------------------------------------------------------------------------
        Args:
            model: Sinerf. Model whose first layer is constrained
            schedule: FrequencySchedule. Schedule of the controlled value
        """
        self.model = model
        self.schedule = schedule
        self.t = 0
        self._apply()

    @property
    def value(self) -> float:
        """Scheduled value at the current step."""
        return self.schedule(self.t)

    @property
    @torch.no_grad()
    def max_frequency(self) -> float:
        """
        Largest angular frequency w_0 * ||w_i|| across first-layer neurons,
        in radians per scene unit.
        """
        W = self.model.first_layer.fc_layer.weight
        return (self.model.omega * W.norm(dim=1).max()).item()

    def step(self) -> None:
        """Advances the curriculum by one step and applies it."""
        self.t += 1
        self._apply()

    @abstractmethod
    def _apply(self) -> None:
        """Applies the curriculum to the model at the current step."""
        pass


class OmegaAnnealer(FrequencyCurriculum):
    """
    Anneals the w_0 hyperparameter of the first SIREN layer along the
    schedule, holding the final value once the schedule ends.

    Since w_0 scales every first-layer neuron at once, changing it rescales
    all input frequencies together, including those already fitted.
    ----------------------------------------------------------------------------
    """

    def _apply(self) -> None:
        """Sets w_0 to the scheduled value."""
        self.model.set_omega(self.value)


class BandwidthCap(FrequencyCurriculum):
    """
    Caps the angular frequency w_0 * ||w_i|| of each first-layer neuron at a
    scheduled cutoff, by projecting rows that exceed it back onto the cutoff
    (direction is kept, only the norm shrinks). Neurons below the cutoff are
    left untouched, so already-fitted low frequencies stay stable as the
    cutoff grows. The cap is released once the schedule ends.
    ----------------------------------------------------------------------------
    """

    @torch.no_grad()
    def _apply(self) -> None:
        """Projects first-layer rows onto the scheduled frequency cutoff."""
        if self.t > self.schedule.T:
            return
        W = self.model.first_layer.fc_layer.weight  # (width, pos_dim)
        freq = self.model.omega * W.norm(dim=1, keepdim=True)
        W.mul_((self.value / freq.clamp_min(1e-12)).clamp(max=1.0))
