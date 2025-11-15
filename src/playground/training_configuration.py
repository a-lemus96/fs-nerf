from dataclasses import dataclass
import torch
from torch import device as Device
from typing import Dict, Any
from argparse import Namespace

from occ_estimator_configuration import OccupancyGridEstimatorConfiguration


@dataclass
class TrainingConfiguration:
    training_device: torch.device
    num_iterations: int
    batch_size: int
    learning_rate: float
    lr_scheduler_type: str
    lr_scheduler_kwargs: Dict[str, Any]
    weight_decay_importance: float
    weight_decay_reg_fn: str
    occupancy_estimator_settings: OccupancyGridEstimatorConfiguration
    white_background: bool

    def __init__(self, training_device: Device, args: Namespace):
        """Builds a NeRF training configuration from an :class:`argparse.Namespace` instance and a :class:`torch.device` instance."""
        try:
            self.training_device = training_device
            self.num_iterations = args.n_iters
            self.batch_size = args.batch_size
            self.learning_rate = args.lro
            self.lr_scheduler_type = args.scheduler
            self.scheduler_kwargs = self.__get_scheduler_kwargs(args)
            self.weight_decay_importance = args.ao
            self.weight_decay_reg_fn = args.reg
            self.white_background = args.white_bkgd
        except KeyError as e:
            raise KeyError(
                f"One or more training parameter keys were not found in input args obj:\n{args}\n\nCheck parser arguments. {e}"
            )

        self.occupancy_estimator_settings = OccupancyGridEstimatorConfiguration()

    def __get_scheduler_kwargs(self, args: Namespace) -> Dict[str, Any]:
        kwargs_dict = {
            "const": {},
            "exp": {"r": args.decay_rate},
        }
        return kwargs_dict[args.scheduler]
