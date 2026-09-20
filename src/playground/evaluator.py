from argparse import Namespace
from dataclasses import dataclass
import math

from lpips import LPIPS
from skimage.metrics import structural_similarity as SSIM
from torch import nn
from torch import device as Device
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb

from render.renderer import Renderer
from utils import load_or_create_config


DEFAULT_EVALUATION_CONFIG_PATH = "../configs/evaluation.yaml"

_DEFAULTS = {
    "val_every": 1000,
}


@dataclass
class EvaluationConfig:
    """
    Holds all hyperparameters required to configure a ModelEvaluator.

    val_every may be overridden from the CLI; when not given (None), it
    falls back to the evaluation YAML config file.

    Fields:
        - training_device (torch.device):   device to run evaluation on
        - val_every (int):                  number of training iterations
                                            between validation steps
    """
    training_device: Device
    val_every: int

    def __init__(
        self,
        training_device: Device,
        val_every: int | None = None,
        config_path: str = DEFAULT_EVALUATION_CONFIG_PATH,
    ):
        """
        Builds an EvaluationConfig from a torch.device instance, a
        CLI-provided validation rate, and the evaluation YAML config file.

        Args:
            training_device (Device): device to run evaluation on
            val_every (int | None): number of training iterations between
                validation steps; CLI-driven, falls back to the YAML config
                file if None
            config_path (str): path to the evaluation YAML config file,
                created with default values if it doesn't exist
        """
        cfg = load_or_create_config(config_path, _DEFAULTS)
        self.training_device = training_device
        self.val_every = val_every if val_every is not None else cfg["val_every"]


class ModelEvaluator:
    """
    Evaluator for NeRF-like models. Uses an occupancy grid estimator to
    accelerate rendering during evaluation.

    Computes PSNR, SSIM, and LPIPS metrics over a full evaluation dataset.
    Iterates directly over dataset tensors — no DataLoader or worker processes.

    Owns its EvaluationConfig (built here, from CLI-provided overrides and
    the YAML config file, and never exposed to callers). Camera intrinsics
    (hwf) are dataset-dependent and aren't known at construction time — the
    caller (ModelTrainer.fit()) provides them afterward via set_hwf().
    """

    def __init__(
        self,
        training_device: Device,
        args: Namespace,
        config_path: str = DEFAULT_EVALUATION_CONFIG_PATH,
    ):
        """
        Args:
            training_device (Device): device to run evaluation on
            args (Namespace): parsed command-line arguments; debug and
                val_every are unpacked from it — val_every falls back to
                the evaluation YAML config file when None
            config_path (str): path to the evaluation YAML config file,
                created with default values if it doesn't exist
        """
        settings = EvaluationConfig(training_device, args.val_every, config_path)
        self.hwf = None
        self.configure(settings, args.debug)

    def configure(self, settings: EvaluationConfig, debug: bool = False):
        """
        Applies an EvaluationConfig to the evaluator.
        Called at construction and can be called again to reconfigure.

        Args:
            settings (EvaluationConfig): full evaluation configuration
            debug (bool): if True, disables all wandb logging
        """
        self._apply_evaluation_config(settings)
        self._lpips_model = self._create_lpips_model()
        self.debug_mode = debug

    def _apply_evaluation_config(self, settings: EvaluationConfig):
        """
        Unpacks scalar hyperparameters from an EvaluationConfig onto
        the evaluator instance.

        Args:
            settings (EvaluationConfig): full evaluation configuration
        """
        self.training_device = settings.training_device
        self.val_every = settings.val_every

    def set_hwf(self, hwf: tuple) -> None:
        """
        Sets the camera intrinsics used for rendering during evaluation.

        Args:
            hwf (tuple): image height, width, and focal length
        """
        self.hwf = hwf

    def _create_lpips_model(self) -> LPIPS:
        """Creates an instance of the :class:`lpips.LPIPS` class. Uses 'vgg' as pretrained backbone model."""
        return LPIPS(net="vgg")

    def evaluate(
        self,
        model: nn.Module,
        renderer: Renderer,
        dataset: Dataset,
        prefix: str = "val",
    ) -> tuple[float, float, float, float]:
        """
        Evaluates the model over the full dataset and returns PSNR, SSIM,
        LPIPS, and their geometric-mean average.

        Unless debug mode is active, the metrics and the rendered RGB and
        depth images are also logged to wandb under f"{prefix}_<name>" keys
        (e.g. val_psnr, val_image, val_depth). They are logged with
        commit=False, so they join the caller's next wandb.log call for the
        current step, or are flushed when the run finishes.

        The dataset should already be on CPU or GPU. No device transfer is
        performed here. The renderer is expected to already be on the
        training device — that part is still the caller's responsibility.
        Train/eval mode is not: model and renderer are temporarily switched
        to eval mode here if they aren't already, and restored to their
        prior mode before returning, so callers don't need to manage this
        around evaluate() calls.

        Args:
            model (nn.Module): trained NeRF-like model
            renderer (Renderer): renderer used to accelerate evaluation
            dataset (Dataset): evaluation dataset
            prefix (str): prefix of the logged wandb keys, e.g. "val" for
                periodic validation or "final" for the end-of-training
                evaluation
        Returns:
            tuple[float, float, float, float]: (psnr, ssim, lpips, average)
        """
        was_model_training = model.training
        was_renderer_training = renderer.training
        if was_model_training:
            model.eval()
        if was_renderer_training:
            renderer.eval()

        H, W, _ = self.hwf
        rgbs_gt = []
        rgbs_predicted = []
        depths_predicted = []

        with torch.no_grad():
            for i in range(len(dataset)):
                rgb_gt = dataset.rgb[i].reshape(H, W, 3)

                rgbs_gt.append(rgb_gt)
                rgb_predicted, depth_predicted = renderer.render_frame_from_rays(
                    dataset.rays_o[i],
                    dataset.rays_d[i],
                    self.hwf,
                    dataset.near,
                    dataset.far,
                    model,
                )
                rgbs_predicted.append(rgb_predicted)
                depths_predicted.append(depth_predicted)

        # Stack and permute to (N, 3, H, W) for metric computation
        rgbs_predicted = torch.permute(torch.stack(rgbs_predicted, dim=0), (0, 3, 1, 2))
        depths_predicted = torch.stack(depths_predicted, dim=0)
        rgbs_gt = torch.permute(torch.stack(rgbs_gt, dim=0), (0, 3, 1, 2))
        rgbs_gt = rgbs_gt.to(self.training_device)

        psnr = self._compute_psnr_metric(rgbs_predicted, rgbs_gt)
        lpips = self._compute_lpips_metric(rgbs_predicted, rgbs_gt)
        ssim = self._compute_ssim_metric(rgbs_predicted, rgbs_gt)
        average = self._compute_average_metric(psnr, ssim, lpips)

        if was_model_training:
            model.train()
        if was_renderer_training:
            renderer.train()

        if not self.debug_mode:
            self._log_to_wandb(
                prefix,
                {"psnr": psnr, "ssim": ssim, "lpips": lpips, "average": average},
                rgbs_predicted,
                depths_predicted,
            )

        return psnr, ssim, lpips, average

    def _log_to_wandb(
        self,
        prefix: str,
        metrics: dict[str, float],
        rgbs_predicted: torch.Tensor,
        depths_predicted: torch.Tensor,
    ) -> None:
        """
        Logs evaluation metrics and the rendered RGB / colorized depth images
        under f"{prefix}_<name>" keys, with commit=False so the values join the
        current wandb step instead of opening a step of their own.

        Args:
            prefix (str): prefix of every logged key
            metrics (dict[str, float]): metric name -> value
            rgbs_predicted (Tensor): (N, 3, H, W) renders in [0, 1]
            depths_predicted (Tensor): (N, H, W) depth maps
        """
        payload = {f"{prefix}_{name}": value for name, value in metrics.items()}
        payload[f"{prefix}_image"] = [
            wandb.Image(
                Renderer.to_uint8_image(rgb.permute(1, 2, 0)),
                caption=f"{prefix} #{i}",
            )
            for i, rgb in enumerate(rgbs_predicted)
        ]
        # depth is normalized per frame, so the caption carries its range
        payload[f"{prefix}_depth"] = [
            wandb.Image(
                Renderer.colorize_depth(depth),
                caption=f"depth [{depth.min().item():.3f}, {depth.max().item():.3f}]",
            )
            for depth in depths_predicted
        ]
        wandb.log(payload, commit=False)

    def _compute_psnr_metric(self, rgbs_predicted: torch.Tensor,
                              rgbs_gt: torch.Tensor) -> float:
        """
        Computes the mean of per-image PSNR values (not the PSNR of the
        pooled MSE across images, which is a different, biased quantity).
        """
        per_image_mse = F.mse_loss(
            rgbs_predicted, rgbs_gt, reduction="none"
        ).mean(dim=(1, 2, 3))
        per_image_psnr = -10.0 * torch.log10(per_image_mse)
        return per_image_psnr.mean().item()

    def _compute_lpips_metric(self, rgbs_predicted: torch.Tensor,
                               rgbs_gt: torch.Tensor) -> float:
        """
        Computes the LPIPS metric on CPU, so the LPIPS model never has to be
        moved to the GPU.

        Renders arrive in [0, 1]; `normalize=True` has the LPIPS model remap
        them to the [-1, 1] range it expects.
        """
        rgbs_predicted = rgbs_predicted.cpu()
        rgbs_gt = rgbs_gt.cpu()
        return self._lpips_model(
            rgbs_predicted, rgbs_gt, normalize=True
        ).mean().item()

    def _compute_ssim_metric(self, rgbs_predicted: torch.Tensor,
                              rgbs_gt: torch.Tensor) -> float:
        """Computes mean SSIM over all images."""
        pred_np = rgbs_predicted.permute(0, 2, 3, 1).cpu().numpy()
        gt_np = rgbs_gt.permute(0, 2, 3, 1).cpu().numpy()
        scores = [
            SSIM(p, g, channel_axis=-1, data_range=1.0)
            for p, g in zip(pred_np, gt_np)
        ]
        return float(sum(scores) / len(scores))

    def _compute_average_metric(self, psnr: float, ssim: float,
                                 lpips: float) -> float:
        """
        Computes the geometric mean of sqrt(MSE), sqrt(1 - SSIM), and LPIPS,
        as used by RegNeRF/FreeNeRF to summarize reconstruction quality.
        """
        mse = 10.0 ** (-psnr / 10.0)
        return (math.sqrt(mse) * math.sqrt(1.0 - ssim) * lpips) ** (1.0 / 3.0)