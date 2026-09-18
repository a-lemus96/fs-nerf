# stdlib modules
from dataclasses import dataclass
from collections.abc import Callable

# third-party modules
import torch
from torch import Tensor
from torch.nn import Module
from nerfacc.estimators.occ_grid import OccGridEstimator

# custom modules
from utils import load_or_create_config, use_generator

DEFAULT_ESTIMATOR_CONFIG_PATH = "../configs/rendering.yaml"

_DEFAULTS = {
    "grid_resolution": 128,
    # TODO: This works for LLFF only, for Blender dataset set to 1
    "grid_num_levels": 4,
    "render_step_size": 5e-3,
    "occ_thre": 1e-2,
    "ema_decay": 0.95,
    "warmup_steps": 256,
    "update_period": 16,
    "early_stop_eps": 1e-4,
    "near_plane": 0.0,
    "far_plane": 1e10,
}


@dataclass
class EstimatorConfig:
    aabb: list[float]  # axis-aligned bounding box
    grid_resolution: int
    grid_num_levels: int
    render_step_size: float
    occ_thre: float          # occupancy threshold for the binary grid
    ema_decay: float         # EMA decay applied to occupancy values
    warmup_steps: int        # steps before grid updates start thresholding
    update_period: int       # grid is refreshed every this many steps
    early_stop_eps: float    # transmittance threshold for ray early-stopping
    near_plane: float        # near plane distance for ray sampling
    far_plane: float         # far plane distance for ray sampling

    def __init__(
        self, aabb: list[float], config_path: str = DEFAULT_ESTIMATOR_CONFIG_PATH
    ):
        """
        Args:
            aabb (list[float]): axis-aligned bounding box; dataset-dependent,
                so it isn't part of the YAML config file
            config_path (str): path to the estimator YAML config file,
                created with default values if it doesn't exist
        """
        cfg = load_or_create_config(config_path, _DEFAULTS)
        self.aabb = aabb
        self.grid_resolution = cfg["grid_resolution"]
        self.grid_num_levels = cfg["grid_num_levels"]
        self.render_step_size = cfg["render_step_size"]
        self.occ_thre = cfg["occ_thre"]
        self.ema_decay = cfg["ema_decay"]
        self.warmup_steps = cfg["warmup_steps"]
        self.update_period = cfg["update_period"]
        self.early_stop_eps = cfg["early_stop_eps"]
        self.near_plane = cfg["near_plane"]
        self.far_plane = cfg["far_plane"]


class OccupancyEstimator:
    """
    Wrapper around nerfacc's OccGridEstimator.

    Owns the grid's configuration (aabb, resolution, levels, and the update
    knobs in EstimatorConfig) so callers deal with a single, project-level
    interface instead of nerfacc's estimator directly. EstimatorConfig is
    constructed here, from the YAML config file, and never exposed to callers.
    """

    def __init__(
        self,
        aabb: list[float],
        seed: int,
        config_path: str = DEFAULT_ESTIMATOR_CONFIG_PATH,
    ) -> None:
        """
        Args:
            aabb (list[float]): axis-aligned bounding box for the grid
            seed (int): seeds a dedicated generator driving sampling
                stratification and grid-update jitter, independent of other
                RNG streams.
            config_path (str): path to the estimator YAML config file,
                created with default values if it doesn't exist
        """
        settings = EstimatorConfig(aabb, config_path)
        self.render_step_size = settings.render_step_size
        self.early_stop_eps = settings.early_stop_eps
        self.occ_thre = settings.occ_thre
        self.ema_decay = settings.ema_decay
        self.warmup_steps = settings.warmup_steps
        self.update_period = settings.update_period
        self.near_plane = settings.near_plane
        self.far_plane = settings.far_plane
        self.__estimator = self.__create_occupancy_estimator(settings)
        self.__seed = seed
        self.__generator = torch.Generator().manual_seed(seed)

    def __create_occupancy_estimator(
        self, settings: EstimatorConfig
    ) -> OccGridEstimator:
        """
        Instantiates an OccGridEstimator from a configuration object.

        Args:
            settings (EstimatorConfig): estimator config
        Returns:
            OccGridEstimator: initialised occupancy grid estimator
        """
        return OccGridEstimator(
            roi_aabb=settings.aabb,
            resolution=settings.grid_resolution,
            levels=settings.grid_num_levels,
        )

    def to(self, device: torch.device) -> None:
        """
        Moves the underlying estimator to the given device.

        Also rebuilds the sampling/update generator on that device. A
        CPU-only generator would silently go unused once training runs on
        CUDA and viceversa.

        Args:
            device (torch.device): target device
        """
        self.__estimator.to(device)
        self.__generator = torch.Generator(device=device).manual_seed(self.__seed)

    def train(self) -> None:
        """Switches the underlying estimator to training mode."""
        self.__estimator.train()

    def eval(self) -> None:
        """Switches the underlying estimator to evaluation mode."""
        self.__estimator.eval()

    def state_dict(self) -> dict:
        """Returns the underlying occupancy grid estimator's state dict."""
        return self.__estimator.state_dict()

    def step(self, step: int, model: Module) -> None:
        """
        Updates the occupancy grid using the current model's density
        predictions. Called at every training iteration.

        Args:
            step (int): current training iteration index
            model (Module): model used to evaluate occupancy
        """

        def occ_eval_fn(x):
            return model(x) * self.render_step_size

        with use_generator(self.__generator):
            self.__estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=self.occ_thre,
                ema_decay=self.ema_decay,
                warmup_steps=self.warmup_steps,
                n=self.update_period,
            )

    def sample(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        sigma_fn: Callable,
        stratified: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Samples points along rays using the occupancy grid.

        Callers toggle between stratified (training) and
        deterministic (eval) sampling.

        Args:
            rays_o (Tensor):        (n_rays, 3) ray origins
            rays_d (Tensor):        (n_rays, 3) ray directions
            sigma_fn (Callable):    density query function
            stratified (bool):      if True, enables stratified sampling
        Returns:
            ray_indices, t_starts, t_ends: packed per-sample outputs
        """
        with use_generator(self.__generator):
            return self.__estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                render_step_size=self.render_step_size,
                early_stop_eps=self.early_stop_eps,
                stratified=stratified,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
            )
