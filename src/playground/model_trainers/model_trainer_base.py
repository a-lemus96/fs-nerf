from abc import ABC, abstractmethod
from nerfdata.datasets import llff


class ModelTrainerBase(ABC):
    """Base class for model trainers."""

    @abstractmethod
    def fit(model: llff.LLFFDataset):
        pass
