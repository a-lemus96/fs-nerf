from dataclasses import dataclass, field
import torch
from torch import device as Device
from typing import Dict, Any, Optional
from argparse import Namespace

from playground.occ_estimator_configuration import OccupancyGridEstimatorConfiguration
from core.occlusion import OcclusionRegularizer


@dataclass
class TrainingConfiguration:
    """
    Holds all hyperparameters and components required to configure a
    NeRFModelTrainer.

    Scalar hyperparameters are parsed from a command-line argparse.Namespace.
    The occlusion regularizer is injected by the caller, keeping the trainer
    decoupled from any specific regularizer implementation.

    Fields:
        - training_device (torch.device):   device to run training on
        - num_iterations (int):             total number of training iterations
        - batch_size (int):                 number of rays per gradient step
        - learning_rate (float):            initial learning rate
        - lr_scheduler_type (str):          one of 'const' or 'exp'
        - lr_scheduler_kwargs (dict):       additional kwargs for the scheduler
        - weight_decay_importance (float):  alpha, importance of freq. regularizer
        - weight_decay_reg_fn (str):        norm type for freq. regularizer ('l1' or 'l2')
        - occupancy_estimator_settings:     config for the occupancy grid estimator
        - white_background (bool):          whether to composite over white background
        - occl_beta (float | None):         importance weight for occlusion regularizer
        - occl_regularizer:                 concrete OcclusionRegularizer, or None
    """
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
    occl_beta: Optional[float]
    occl_regularizer: Optional[OcclusionRegularizer]

    def __init__(
        self,
        training_device: Device,
        args: Namespace,
        occl_regularizer: Optional[OcclusionRegularizer] = None,
    ):
        """
        Builds a TrainingConfiguration from a parsed argument namespace and an
        optional occlusion regularizer instance.

        The caller is responsible for constructing the concrete regularizer and
        passing it here. Passing None disables occlusion regularization entirely.

        Args:
            training_device (Device):            device to run training on
            args (Namespace):                    parsed command-line arguments
            occl_regularizer (OcclusionRegularizer | None):
                                                 concrete regularizer instance,
                                                 or None to disable
        Raises:
            KeyError: if a required argument key is missing from args
        """
        try:
            self.training_device = training_device
            self.num_iterations = args.n_iters
            self.batch_size = args.batch_size
            self.learning_rate = args.lro
            self.lr_scheduler_type = args.scheduler
            self.lr_scheduler_kwargs = self.__get_scheduler_kwargs(args)
            self.weight_decay_importance = args.ao
            self.weight_decay_reg_fn = args.reg
            self.white_background = args.white_bkgd
            self.occl_beta = args.beta
        except KeyError as e:
            raise KeyError(
                f"One or more training parameter keys were not found in input "
                f"args obj:\n{args}\n\nCheck parser arguments. {e}"
            )

        self.occl_regularizer = occl_regularizer
        self.occupancy_estimator_settings = OccupancyGridEstimatorConfiguration()

    def __get_scheduler_kwargs(self, args: Namespace) -> Dict[str, Any]:
        """
        Extracts the keyword arguments required by the chosen learning rate
        scheduler from the argument namespace.
        ------------------------------------------------------------------------
        Args:
            args (Namespace): parsed command-line arguments
        Returns:
            Dict[str, Any]: keyword arguments for the scheduler constructor
        """
        kwargs_dict = {
            "const": {},
            "exp": {"r": args.decay_rate},
        }
        return kwargs_dict[args.scheduler]