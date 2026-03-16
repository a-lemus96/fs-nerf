from nerfacc.estimators.occ_grid import OccGridEstimator
from lpips import LPIPS
from skimage.metrics import structural_similarity as SSIM
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from playground.model_evaluators.model_evaluator_base import ModelEvaluatorBase
from playground.configuration.evaluation_configuration import EvaluationConfiguration
import render.rendering as R


class NeRFModelEvaluator(ModelEvaluatorBase):
    """
    Evaluator for NeRF-like models. Uses an occupancy grid estimator to
    accelerate rendering during evaluation.

    Computes PSNR, SSIM, and LPIPS metrics over a full evaluation dataset.
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
        self._lpips_model = self._create_lpips_model()
        self.white_background = settings.white_background
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

        Args:
            model (nn.Module): trained NeRF-like model
            estimator (OccGridEstimator): occupancy grid estimator
            dataset (Dataset): evaluation dataset
        Returns:
            Tuple[float, float, float]: (psnr, ssim, lpips)
        """
        rgbs_gt = []
        rgbs_predicted = []
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

        with torch.no_grad():
            for sample in data_loader:
                rgb_gt, pose = sample
                rgbs_gt.append(rgb_gt)
                rgb_predicted, _ = R.render_frame(
                    self.hwf,
                    data_loader.dataset.near,
                    data_loader.dataset.far,
                    pose[0],
                    self.chunk_size,
                    estimator,
                    model,
                    train=False,
                    ndc=dataset.ndc,
                    white_bkgd=self.white_background,
                    render_step_size=5e-3,
                    device=self.training_device,
                )
                rgbs_predicted.append(rgb_predicted)

        rgbs_predicted = torch.permute(torch.stack(rgbs_predicted, dim=0), (0, 3, 1, 2))
        rgbs_gt = torch.permute(torch.cat(rgbs_gt, dim=0), (0, 3, 1, 2))
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
        val_size = rgbs_predicted.shape[0]
        self._lpips_model.to(self.training_device)
        rgbs_predicted = rgbs_predicted.to(self.training_device)

        with torch.no_grad():
            if val_size <= self.lpips_chunk_size:
                return self._lpips_model(rgbs_predicted, rgbs_gt).mean().item()

            val_lpips = 0.0
            n_chunks = 0
            for i in range(0, val_size, self.lpips_chunk_size):
                chunk = rgbs_predicted[i:i + self.lpips_chunk_size]
                chunk_gt = rgbs_gt[i:i + self.lpips_chunk_size]
                val_lpips += self._lpips_model(chunk, chunk_gt).mean().item()
                n_chunks += 1
            return val_lpips / n_chunks

    def _compute_ssim_metric(self, rgbs_predicted: torch.Tensor,
                              rgbs_gt: torch.Tensor) -> float:
        """Computes structural similarity index. Computation is performed on CPU."""
        rgbs_predicted = torch.permute(rgbs_predicted, (0, 2, 3, 1)).cpu().numpy()
        rgbs_gt = torch.permute(rgbs_gt, (0, 2, 3, 1)).cpu().numpy()

        ssim = 0.
        for rgb_predicted, rgb_gt in zip(rgbs_predicted, rgbs_gt):
            ssim += SSIM(rgb_predicted, rgb_gt, channel_axis=-1,
                         data_range=1.0, gaussian_weights=True)
        return ssim / len(rgbs_predicted)