from abc import ABC, abstractmethod
from nerfdata.datasets import llff


class ModelEvaluator(ABC):
    """Base class for model evaluators."""

    @abstractmethod
    def evaluate(model):
        pass
