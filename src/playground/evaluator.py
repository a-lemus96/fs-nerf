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

import render.rendering as R
from playground.estimator import OccupancyEstimator
from utils import load_or_create_config


DEFAULT_EVALUATION_CONFIG_PATH = "../configs/evaluation.yaml"

_DEFAULTS = {
    "chunk_size": 1024,  # rays per rendering chunk during validation/evaluation
    "val_every": 1000,
}


@dataclass
class EvaluationConfig:
    """
    Holds all hyperparameters required to configure a ModelEvaluator.

    val_every may be overridden from the CLI; when not given (None), it
    falls back to the evaluation YAML config file, as does chunk_size.

    Fields:
        - training_device (torch.device):   device to run evaluation on
        - chunk_size (int):                 number of rays per rendering chunk
        - val_every (int):                  number of training iterations
                                            between validation steps
    """
    training_device: Device
    chunk_size: int
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
        self.chunk_size = cfg["chunk_size"]
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
        self.chunk_size = settings.chunk_size
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

    def evaluate(self, model: nn.Module, estimator: OccupancyEstimator,
                 dataset: Dataset) -> tuple[float, float, float, float]:
        """
        Evaluates the model over the full dataset and returns PSNR, SSIM,
        LPIPS, and their geometric-mean average.

        Iterates directly over dataset.imgs and dataset.poses tensors, avoiding
        DataLoader and worker process overhead. The dataset should already be on
        CPU or GPU — no device transfer is performed here.

        Args:
            model (nn.Module): trained NeRF-like model
            estimator (OccupancyEstimator): occupancy grid estimator
            dataset (Dataset): evaluation dataset (img_mode=True)
        Returns:
            tuple[float, float, float, float]: (psnr, ssim, lpips, average)
        """
        rgbs_gt = []
        rgbs_predicted = []

        with torch.no_grad():
            for i in range(len(dataset.imgs)):
                rgb_gt = dataset.imgs[i]       # (H, W, 3)
                pose = dataset.poses[i]        # (3, 4)

                rgbs_gt.append(rgb_gt)
                rgb_predicted, _ = R.render_frame(
                    self.hwf,
                    dataset.near,
                    dataset.far,
                    pose,
                    self.chunk_size,
                    estimator,
                    model,
                    train=False,
                    ndc=dataset.ndc,
                    render_step_size=estimator.render_step_size,
                    early_stop_eps=estimator.early_stop_eps,
                    device=self.training_device,
                )
                rgbs_predicted.append(rgb_predicted)

        # Stack and permute to (N, 3, H, W) for metric computation
        rgbs_predicted = torch.permute(torch.stack(rgbs_predicted, dim=0), (0, 3, 1, 2))
        rgbs_gt = torch.permute(torch.stack(rgbs_gt, dim=0), (0, 3, 1, 2))
        rgbs_gt = rgbs_gt.to(self.training_device)

        psnr = self._compute_psnr_metric(rgbs_predicted, rgbs_gt)
        lpips = self._compute_lpips_metric(rgbs_predicted, rgbs_gt)
        ssim = self._compute_ssim_metric(rgbs_predicted, rgbs_gt)
        average = self._compute_average_metric(psnr, ssim, lpips)

        return psnr, ssim, lpips, average

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