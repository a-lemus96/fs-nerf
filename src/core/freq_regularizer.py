from abc import ABC, abstractmethod
from collections.abc import Iterator

from torch.nn.parameter import Parameter
from torch import Tensor
import torch


class FrequencyScheduler(ABC):
    """
    Abstract base class for frequency schedulers.

    A frequency scheduler defines how the regularization importance parameter
    alpha varies over training iterations. Subclasses must implement the alpha
    property, which returns the current regularization weight as a function
    of the current training step.
    """

    def __init__(self) -> None:
        """Initializes the scheduler at step zero."""
        self.current_step = 0

    def step(self) -> None:
        """Advances the scheduler by one training iteration."""
        self.current_step += 1

    @property
    @abstractmethod
    def alpha(self) -> float:
        """Returns the regularization weight alpha at the current training step."""
        pass


class ConstantScheduler(FrequencyScheduler):
    """
    Constant frequency scheduler.

    Returns a fixed regularization weight alpha at every training iteration.
    This is equivalent to standard static frequency regularization, and
    serves as a baseline against dynamic scheduling strategies.
    """

    def __init__(self, alpha: float) -> None:
        """
        Initializes the constant scheduler.
        ------------------------------------------------------------------------
        Args:
            alpha: float. Fixed regularization weight returned at every step.
        """
        super().__init__()
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """Returns the fixed regularization weight α."""
        return self._alpha


class LinearScheduler(FrequencyScheduler):
    """
    Linear frequency scheduler.

    Linearly interpolates the regularization weight alpha from an initial value
    alpha_0 to a final value alpha_T over T training iterations. After T
    iterations, alpha_T is returned at every subsequent step.
    """

    def __init__(self, alpha_0: float, alpha_T: float, T: int) -> None:
        """
        Initializes the linear scheduler.
        ------------------------------------------------------------------------
        Args:
            alpha_0: float. Initial regularization weight at step 0.
            alpha_T: float. Final regularization weight at step T.
            T: int. Number of iterations over which to interpolate.
        """
        super().__init__()
        self.alpha_0 = alpha_0
        self.alpha_T = alpha_T
        self.T = T

    @property
    def alpha(self) -> float:
        """
        Returns the linearly interpolated regularization weight alpha at the
        current training step.
        """
        t = min(self.current_step, self.T)
        return self.alpha_0 + (self.alpha_T - self.alpha_0) * (t / self.T)


class FrequencyRegularizer:
    """
    Frequency regularizer for SIREN-based NeRF models.

    Penalizes the magnitude of the model's weight parameters, which in a SIREN
    loosely corresponds to constraining the angular frequencies of the sinusoidal
    activations. The penalization importance alpha is controlled by a FrequencyScheduler,
    allowing the regularization strength to vary over training iterations.

    This promotes low-frequency solutions early in training, while gradually
    allowing higher-frequency components to emerge as alpha decays.
    """

    def __init__(
        self,
        model_parameters: Iterator[tuple[str, Parameter]],
        scheduler: FrequencyScheduler,
        reg: str = "l1",
    ) -> None:
        """
        Initializes the frequency regularizer.

        Args:
            model_parameters: Iterator[tuple[str, Parameter]]. Named parameters
                of the model to regularize, as returned by model.named_parameters().
            scheduler: FrequencyScheduler. Schedule defining how alpha evolves over
                training iterations.
            reg: str. Norm used to penalize weights. One of 'l1' or 'l2'.
        Raises:
            ValueError: if reg is not 'l1' or 'l2'.
        """
        self.model_parameters = self._capture_model_parameters(model_parameters)
        self.freq_scheduler = scheduler
        if reg not in ("l1", "l2"):
            raise ValueError(f"reg must be 'l1' or 'l2', got '{reg}'")
        self.reg = reg

    def _capture_model_parameters(
        self, model_parameters: Iterator[tuple[str, Parameter]]
    ) -> tuple[tuple[str, Parameter], ...]:
        """
        Filters and captures the relevant model parameters into a
        non-consumable container.

        Only weight parameters with more than 3 output features are retained,
        excluding small output layers such as the RGB and sigma heads.

        Args:
            model_parameters: Iterator[tuple[str, Parameter]]. Named parameters
                of the model, as returned by model.named_parameters().
        Returns:
            tuple[tuple[str, Parameter], ...]. Filtered named parameters.
        """
        return tuple(
            (name, param)
            for name, param in model_parameters
            if "weight" in name and param.shape[0] > 3
        )

    def step(self) -> None:
        """Advances the frequency scheduler by one training iteration."""
        self.freq_scheduler.step()

    def __call__(self) -> Tensor:
        """
        Computes the weighted frequency regularization loss at the current
        training step.

        Applies the configured norm to all captured weight parameters and
        scales the result by the current scheduler weight alpha.

        Returns:
            Tensor: scalar regularization loss alpha(k) · Omega(w).
        """
        alpha = self.freq_scheduler.alpha
        reg = torch.stack(
            [
                torch.abs(p).sum() if self.reg == "l1" else torch.square(p).sum()
                for _, p in self.model_parameters
            ]
        ).sum()
        return alpha * reg
