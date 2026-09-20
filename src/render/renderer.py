# stdlib modules
from dataclasses import dataclass

# third-party modules
from nerfacc.volrend import rendering, render_weight_from_density
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
from torch import nn
from torch import Tensor
from tqdm import tqdm

# custom modules
import utils.utilities as U
from playground.estimator import OccupancyEstimator, _DEFAULTS as _ESTIMATOR_DEFAULTS
from utils import load_or_create_config


# Function to map float values to [0, 255] integer range
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

DEFAULT_RENDERING_CONFIG_PATH = "../configs/rendering.yaml"

_DEFAULTS = {
    "chunk_size": 1024,  # rays per rendering chunk
}

# rendering.yaml is shared with EstimatorConfig, so file-creation must write
# every key either config might read, regardless of which one runs first.
_FULL_DEFAULTS = {**_ESTIMATOR_DEFAULTS, **_DEFAULTS}


@dataclass
class RenderingResult:
    """
    Holds per-ray and per-sample outputs from a volumetric rendering pass.

    Per-ray fields (required, n_rays = number of rays in batch):
        - rgb         (n_rays, 3): rendered RGB color per ray
        - depth       (n_rays, 1): rendered depth per ray
        - opacity     (n_rays, 1): accumulated opacity per ray
        - n_rays      (int):       total number of rays in the batch

    Per-sample fields (optional, S = total samples across all rays):
        - weights     (S,): rendering weights w_k = T_k * (1 - exp(-sigma_k * delta_k))
        - t_vals      (S,): midpoint depth values (in NDC space if ndc=True)
        - ray_indices (S,): index of the ray each sample belongs to

    Per-sample fields are always populated by render_rays but typed as
    Optional to reflect that regularizers should guard against the empty-
    tensor case that arises when the occupancy estimator produces no samples.
    """
    rgb:         Tensor
    depth:       Tensor
    opacity:     Tensor
    n_rays:      int
    weights:     Tensor | None = None
    t_vals:      Tensor | None = None
    ray_indices: Tensor | None = None


@dataclass
class RenderingConfig:
    chunk_size: int  # rays per rendering chunk

    def __init__(self, config_path: str = DEFAULT_RENDERING_CONFIG_PATH):
        """
        Args:
            config_path (str): path to the shared rendering YAML config file,
                created with default values if it doesn't exist. Also read by
                EstimatorConfig, so defaults must cover both configs' keys.
        """
        cfg = load_or_create_config(config_path, _FULL_DEFAULTS)
        self.chunk_size = cfg["chunk_size"]


class Renderer:
    """
    Encapsulates ray/frame/path rendering for a NeRF-like model and owns the
    occupancy grid estimator used to accelerate it.

    RenderingConfig and the estimator's own EstimatorConfig are both read
    from the same YAML file (rendering.yaml), so a single config file governs
    rendering batch size and occupancy grid behavior together. Callers interact
    with rendering and the occupancy grid only through this class.
    """

    def __init__(
        self,
        aabb: list[float],
        seed: int,
        config_path: str = DEFAULT_RENDERING_CONFIG_PATH,
    ) -> None:
        """
        Args:
            aabb (list[float]): axis-aligned bounding box, passed through to
                the occupancy estimator
            seed (int): seeds the occupancy estimator's dedicated generator
            config_path (str): path to the shared rendering YAML config file,
                created with default values if it doesn't exist
        """
        settings = RenderingConfig(config_path)
        self.chunk_size = settings.chunk_size
        self.__estimator = OccupancyEstimator(aabb, seed, config_path)
        self.device = torch.device("cpu")
        self.training = False

    def to(self, device: torch.device) -> None:
        """
        Moves the renderer (and its occupancy estimator) to the given device.

        Args:
            device (torch.device): target device
        """
        self.device = device
        self.__estimator.to(device)

    def train(self) -> None:
        """Switches the renderer and its occupancy estimator to training mode."""
        self.training = True
        self.__estimator.train()

    def eval(self) -> None:
        """Switches the renderer and its occupancy estimator to evaluation mode."""
        self.training = False
        self.__estimator.eval()

    def step(self, step: int, model: nn.Module) -> None:
        """
        Updates the occupancy grid using the current model's density
        predictions. Called at every training iteration.

        Args:
            step (int): current training iteration index
            model (nn.Module): model used to evaluate occupancy
        """
        self.__estimator.step(step, model)

    def state_dict(self) -> dict:
        """Returns the underlying occupancy grid estimator's state dict."""
        return self.__estimator.state_dict()

    def render_rays(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        model: nn.Module,
    ) -> RenderingResult:
        """
        Renders a batch of rays through a NeRF model using occupancy-grid-
        accelerated sampling and volume rendering.

        Sampling is performed in two passes: a density-only pass to query the
        occupancy estimator, followed by a full RGB+density pass for volume
        rendering. Rendering weights are computed in a third no-grad pass so
        that they are available to occlusion regularizers without affecting
        the computational graph.

        If the occupancy estimator produces no samples (AssertionError from
        nerfacc), a fallback RenderingResult is returned with the background
        color, zero depth and opacity, and empty per-sample tensors.

        Stratified sampling and gradient-requiring background are enabled
        when the renderer is in training mode (see train()/eval()).

        Args:
            rays_o (Tensor): (n_rays, 3) ray origins in world/NDC space
            rays_d (Tensor): (n_rays, 3) ray directions in world/NDC space
            model (nn.Module): NeRF-like model returning (rgb, sigma) or sigma
        Returns:
            RenderingResult:
                rgb         (n_rays, 3): rendered color per ray
                depth       (n_rays, 1): rendered depth per ray
                opacity     (n_rays, 1): accumulated opacity per ray
                n_rays      (int):       number of rays in the batch
                weights     (S,):        per-sample rendering weights (None if empty)
                t_vals      (S,):        per-sample midpoint depths (None if empty)
                ray_indices (S,):        per-sample ray indices (None if empty)
        """
        device = self.device
        train = self.training
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        n_rays = len(rays_o)

        def sigma_fn(t_starts, t_ends, ray_indices):
            to = rays_o[ray_indices]
            td = rays_d[ray_indices]
            x = to + td * (t_starts + t_ends)[:, None] / 2.0
            sigmas = model(x)
            return sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = self.__estimator.sample(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            stratified=train,
        )

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            to = rays_o[ray_indices]
            td = rays_d[ray_indices]
            x = to + td * (t_starts + t_ends)[:, None] / 2.0
            out = model(x, td)
            rgbs = out[..., :3]
            sigmas = out[..., -1]
            return rgbs, sigmas.squeeze(-1)

        render_bkgd = torch.zeros((3,), device=device, requires_grad=train)

        try:
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=n_rays,
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            weights = extras["weights"]

        except AssertionError:
            # occupancy estimator found no samples; return background fallback
            rgb     = torch.zeros_like(rays_o)
            opacity = torch.zeros(n_rays, 1, device=device)
            depth   = torch.zeros(n_rays, 1, device=device)
            weights = torch.zeros(0, device=device)

        t_vals = (t_starts + t_ends) / 2.0

        return RenderingResult(
            rgb=rgb,
            depth=depth,
            opacity=opacity,
            n_rays=n_rays,
            weights=weights,
            t_vals=t_vals,
            ray_indices=ray_indices,
        )

    def render_frame_from_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        hwf: tuple[int, int, float],
        near: float,
        far: float,
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Renders a single image from a flattened, already-cast set of rays by
        chunkifying them to avoid memory issues.

        Rays are split into chunks of chunk_size and rendered independently.
        The resulting RGB and depth chunks are concatenated and reshaped into
        image dimensions. Unlike render_frame, rays are used as given — no
        per-pose ray casting or NDC conversion is performed, so callers that
        already have flattened rays (e.g. a precomputed dataset) can skip
        recomputing them.
        ----------------------------------------------------------------------------
        Args:
            rays_o (Tensor):               (H*W, 3) flattened ray origins
            rays_d (Tensor):               (H*W, 3) flattened ray directions
            hwf (tuple[int, int, float]): camera intrinsics (height, width, focal),
                                           used only to reshape the output
            near (float):                 near depth bound for depth clamping
            far (float):                  far depth bound for depth clamping
            model (nn.Module):            NeRF-like model
        Returns:
            img (Tensor):       (H, W, 3) rendered RGB image
            depth_map (Tensor): (H, W) rendered depth map, clamped to [near, far]
        ----------------------------------------------------------------------------
        """
        H, W, _ = hwf
        chunked_rays_o = U.get_chunks(rays_o, chunksize=self.chunk_size)
        chunked_rays_d = U.get_chunks(rays_d, chunksize=self.chunk_size)

        img = []
        depth_map = []
        for chunk_rays_o, chunk_rays_d in zip(chunked_rays_o, chunked_rays_d):
            out = self.render_rays(
                rays_o=chunk_rays_o,
                rays_d=chunk_rays_d,
                model=model,
            )
            img.append(out.rgb)
            depth_map.append(out.depth)

        img = torch.cat(img, dim=0)
        depth = torch.cat(depth_map, dim=0).clamp(near, far)

        return img.reshape(H, W, 3), depth.reshape(H, W)

    def render_frame(
        self,
        hwf: tuple[int, int, float],
        near: float,
        far: float,
        pose: torch.Tensor,
        model: nn.Module,
        ndc: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Renders a single image from a given camera pose by casting its rays
        and delegating to render_frame_from_rays.

        Rays are cast from the pose and optionally converted to NDC before
        chunked rendering.
        ----------------------------------------------------------------------------
        Args:
            hwf (tuple[int, int, float]): camera intrinsics (height, width, focal)
            near (float):                 near depth bound for depth clamping
            far (float):                  far depth bound for depth clamping
            pose (Tensor):                (4, 4) camera-to-world pose matrix
            model (nn.Module):            NeRF-like model
            ndc (bool):                   if True, converts rays to NDC before rendering
        Returns:
            img (Tensor):       (H, W, 3) rendered RGB image
            depth_map (Tensor): (H, W) rendered depth map, clamped to [near, far]
        ----------------------------------------------------------------------------
        """
        rays_o, rays_d = U.get_rays(pose, hwf, self.device)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        if ndc:
            rays_o, rays_d = U.to_ndc(rays_o, rays_d, hwf, 1.0)

        return self.render_frame_from_rays(
            rays_o,
            rays_d,
            hwf,
            near,
            far,
            model,
        )

    def render_path(
        self,
        render_poses: torch.Tensor,
        hwf: tuple[int, int, float],
        near: float,
        far: float,
        model: nn.Module,
        ndc: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Renders a sequence of frames from a trajectory of camera poses,
        returning RGB and depth frame stacks suitable for video export.

        Each pose is rendered independently using render_frame. Rendering is
        performed under torch.no_grad() since this function is used for
        inference only.
        ----------------------------------------------------------------------------
        Args:
            render_poses (Tensor):        (N, 4, 4) trajectory of camera poses
            hwf (tuple[int, int, float]): camera intrinsics (height, width, focal)
            near (float):                 near depth bound for depth clamping
            far (float):                  far depth bound for depth clamping
            model (nn.Module):            NeRF-like model
            ndc (bool):                   if True, converts rays to NDC before rendering
        Returns:
            frames (ndarray):   (N, H, W, 3) rendered RGB frames in float [0, 1]
            d_frames (ndarray): (N, H, W) rendered depth frames
        ----------------------------------------------------------------------------
        """
        H, W, _ = hwf
        frames, d_frames = [], []
        pbar = tqdm(render_poses, desc="[Rendering Frames]")
        for i, pose in enumerate(pbar):
            with torch.no_grad():
                rgb, depth = self.render_frame(
                    hwf,
                    near,
                    far,
                    pose,
                    model,
                    ndc=ndc,
                )
                rgb   = rgb.reshape([H, W, 3]).detach().cpu().numpy()
                depth = depth.reshape([H, W]).detach().cpu().numpy()

            frames.append(rgb)
            d_frames.append(depth)

        frames   = np.stack(frames, 0)
        d_frames = np.stack(d_frames, 0)

        return frames, d_frames

    def render_video(
        self,
        frames: np.ndarray,
        d_frames: np.ndarray,
        cmap: str = "plasma",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts float RGB and depth frame stacks into uint8 arrays ready for
        video export, applying a colormap to the depth frames.

        Args:
            frames (ndarray):   (N, H, W, 3) RGB frames in float [0, 1]
            d_frames (ndarray): (N, H, W) depth frames
            cmap (str):         matplotlib colormap name applied to depth frames
        Returns:
            rgb_video (ndarray):   (N, 3, H, W) uint8 RGB frames
            depth_video (ndarray): (N, 3, H, W) uint8 colorized depth frames

        """
        norm   = matplotlib.colors.Normalize(vmin=np.amin(d_frames), vmax=np.amax(d_frames))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        d_rgba = mapper.to_rgba(d_frames.flatten())
        d_rgba = np.reshape(d_rgba, list(d_frames.shape) + [4])

        return (
            np.transpose(to8b(frames), (0, 3, 1, 2)),
            np.transpose(to8b(d_rgba[..., :3]), (0, 3, 1, 2)),
        )

    @staticmethod
    def colorize_depth(
        depth: Tensor | np.ndarray, cmap: str = "plasma"
    ) -> np.ndarray:
        """
        Maps a single depth map to a colorized uint8 image, min-max normalized
        over the map itself. Uses the same default colormap as render_video.

        Args:
            depth (Tensor | ndarray): (H, W) depth map
            cmap (str):               matplotlib colormap name
        Returns:
            ndarray: (H, W, 3) uint8 RGB image
        """
        if isinstance(depth, Tensor):
            depth = depth.detach().cpu().numpy()

        norm = matplotlib.colors.Normalize(vmin=depth.min(), vmax=depth.max())
        rgba = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(depth, bytes=True)

        return rgba[..., :3]

    @staticmethod
    def to_uint8_image(image: Tensor | np.ndarray) -> np.ndarray:
        """
        Converts an image into a host-side uint8 array: a tensor is detached
        and moved to the CPU, and a float image in [0, 1] is scaled to
        [0, 255] (values outside [0, 1] are clipped). A uint8 array is
        returned as is.

        Args:
            image (Tensor | ndarray): image, either float in [0, 1] or uint8
        Returns:
            ndarray: uint8 image, same shape as the input
        """
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype != np.uint8:
            image = to8b(image)

        return image