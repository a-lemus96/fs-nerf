from dataclasses import dataclass
from typing import Tuple
from argparse import Namespace
import torch
from torch import device as Device


@dataclass
class EvaluationConfiguration:
    """
    Holds all hyperparameters required to configure a NeRFModelEvaluator.

    Scalar hyperparameters are parsed from a command-line argparse.Namespace.

    Fields:
        - training_device (torch.device):   device to run evaluation on
        - hwf (Tuple):                      camera intrinsics (height, width, focal)
        - chunk_size (int):                 number of rays per rendering chunk
    """
    training_device: Device
    hwf: Tuple
    chunk_size: int
    lpips_chunk_size: int

    def __init__(self, training_device: Device, hwf: Tuple, args: Namespace):
        """
        Builds an EvaluationConfiguration from a parsed argument namespace,
        a torch.device instance, and camera intrinsics.
        """
        self.training_device = training_device
        self.hwf = hwf
        self.chunk_size = args.val_batch_size_multiplier * args.batch_size
        self.lpips_chunk_size = args.lpips_chunk_size