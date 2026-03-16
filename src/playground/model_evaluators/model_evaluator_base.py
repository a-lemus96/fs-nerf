from abc import ABC, abstractmethod
from torch import nn
from torch.utils.data import Dataset
from typing import Tuple


class ModelEvaluatorBase(ABC):
    """Base class for model evaluators."""

    @abstractmethod
    def evaluate(self, model: nn.Module, dataset: Dataset) -> Tuple[float, float, float]:
        pass