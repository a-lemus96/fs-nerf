from abc import ABC
from dataclasses import dataclass
import torch
from torch import device as Device
from typing import Any, Dict
from argparse import Namespace

from playground.configuration.lpips_configuration import 

@dataclass
class EvaluationConfiguration:
    training_device: Device
    num_imgs: int
    batch_size: int
    chunk_size: int
    white_background: bool
