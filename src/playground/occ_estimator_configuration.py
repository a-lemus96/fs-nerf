from dataclasses import dataclass
from typing import List

from argparse import Namespace


@dataclass
class OccupancyGridEstimatorConfiguration:
    aabb: List[float]  # axis-aligned bounding box
    grid_resolution: int
    grid_num_levels: int
    render_step_size = float

    def __init__(self):
        # TODO: aabb depends on the dataset itself, pull it from the dataset
        self.grid_resolution = 128
        # TODO: This works for LLFF only, for Blender dataset set to 1
        self.grid_num_levels = 4
        self.render_step_size = 5e-3
