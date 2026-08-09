from abc import ABC, abstractmethod
from nerfdata import LLFFDataset


class ModelTrainerBase(ABC):
    """Base class for model trainers."""

    @abstractmethod
    def fit(model: LLFFDataset):
        pass
