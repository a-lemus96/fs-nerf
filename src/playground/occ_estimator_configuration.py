from dataclasses import dataclass
from typing import List


@dataclass
class OccupancyGridEstimatorConfiguration:
    aabb: List[float]  # axis-aligned bounding box
    grid_resolution: int
    grid_num_levels: int
    render_step_size: float
    occ_thre: float          # occupancy threshold for the binary grid
    ema_decay: float         # EMA decay applied to occupancy values
    warmup_steps: int        # steps before grid updates start thresholding
    update_period: int       # grid is refreshed every this many steps
    early_stop_eps: float    # transmittance threshold for ray early-stopping

    def __init__(self):
        # TODO: aabb depends on the dataset itself, pull it from the dataset
        self.grid_resolution = 128
        # TODO: This works for LLFF only, for Blender dataset set to 1
        self.grid_num_levels = 4
        self.render_step_size = 5e-3
        self.occ_thre = 1e-2
        self.ema_decay = 0.95
        self.warmup_steps = 256
        self.update_period = 16
        self.early_stop_eps = 1e-4
