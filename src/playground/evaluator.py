from argparse import Namespace
from dataclasses import dataclass

from nerfacc.estimators.occ_grid import OccGridEstimator
from lpips import LPIPS
from skimage.metrics import structural_similarity as SSIM
from torch import nn
from torch import device as Device
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple

import render.rendering as R


@dataclass
class EvaluationConfiguration:
    """
    Holds all hyperparameters required to configure a ModelEvaluator.

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


class ModelEvaluator:
    """
    Evaluator for NeRF-like models. Uses an occupancy grid estimator to
    accelerate rendering during evaluation.

    Computes PSNR, SSIM, and LPIPS metrics over a full evaluation dataset.
    Iterates directly over dataset tensors — no DataLoader or worker processes.
    """

    def __init__(self, settings: EvaluationConfiguration, debug: bool = False):
        self.configure(settings, debug)

    def configure(self, settings: EvaluationConfiguration, debug: bool = False):
        """
        Applies an EvaluationConfiguration to the evaluator.
        Called at construction and can be called again to reconfigure.

        Args:
            settings (EvaluationConfiguration): full evaluation configuration
            debug (bool): if True, disables all wandb logging
        """
        self._apply_evaluation_config(settings)
        self._lpips_model = self._create_lpips_model().to(self.training_device)
        self.debug_mode = debug

    def _apply_evaluation_config(self, settings: EvaluationConfiguration):
        """
        Unpacks scalar hyperparameters from an EvaluationConfiguration onto
        the evaluator instance.

        Args:
            settings (EvaluationConfiguration): full evaluation configuration
        """
        self.training_device = settings.training_device
        self.hwf = settings.hwf
        self.chunk_size = settings.chunk_size
        self.lpips_chunk_size = settings.lpips_chunk_size

    def _create_lpips_model(self) -> LPIPS:
        """Creates an instance of the :class:`lpips.LPIPS` class. Uses 'vgg' as pretrained backbone model."""
        return LPIPS(net="vgg")

    def evaluate(self, model: nn.Module, estimator: OccGridEstimator,
                 dataset: Dataset) -> Tuple[float, float, float]:
        """
        Evaluates the model over the full dataset and returns PSNR, SSIM, and LPIPS.

        Iterates directly over dataset.imgs and dataset.poses tensors, avoiding
        DataLoader and worker process overhead. The dataset should already be on
        CPU or GPU — no device transfer is performed here.

        Args:
            model (nn.Module): trained NeRF-like model
            estimator (OccGridEstimator): occupancy grid estimator
            dataset (Dataset): evaluation dataset (img_mode=True)
        Returns:
            Tuple[float, float, float]: (psnr, ssim, lpips)
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
                    render_step_size=5e-3,
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

        return psnr, ssim, lpips

    def _compute_psnr_metric(self, rgbs_predicted: torch.Tensor,
                              rgbs_gt: torch.Tensor) -> float:
        """Computes peak signal-to-noise ratio."""
        return -10.0 * torch.log10(F.mse_loss(rgbs_predicted, rgbs_gt)).item()

    def _compute_lpips_metric(self, rgbs_predicted: torch.Tensor,
                               rgbs_gt: torch.Tensor) -> float:
        """
        Computes the LPIPS metric. Images are processed in chunks of
        lpips_chunk_size to avoid OOM errors on large datasets.
        """
        n = rgbs_predicted.shape[0]
        lpips_scores = []
        for start in range(0, n, self.lpips_chunk_size):
            end = min(start + self.lpips_chunk_size, n)
            pred_chunk = rgbs_predicted[start:end].to(self.training_device)
            gt_chunk = rgbs_gt[start:end].to(self.training_device)
            lpips_scores.append(self._lpips_model(pred_chunk, gt_chunk).mean())
        return torch.stack(lpips_scores).mean().item()

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