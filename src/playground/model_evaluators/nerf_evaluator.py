from nerfacc.estimators.occ_grid import OccGridEstimator
from lpips import LPIPS
from skimage.metrics import structural_similarity as SSIM
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from playground.model_evaluators.model_evaluator_base import ModelEvaluatorBase
from playground.configuration.evaluation_configuration import EvaluationConfiguration
import render.rendering as R


class NeRFModelEvaluator(ModelEvaluatorBase):
    """Evaluator ADT for NeRF-like models. Uses an occupancy grid estimator. It depends on the
    following elements:
    - NeRF model
    - Occupancy grid estimator"""

    def __init__(self, settings: EvaluationConfiguration, debug: bool = False):
        self.configure(settings, debug)

    def configure(self, settings: EvaluationConfiguration, debug: bool = False):
        """Extracts individual configuration fields and applies them to the evaluator."""
        self._apply_evaluation_config(settings)
        self._lpips_model = self._create_lpips_model()
        self.white_background = settings.white_background

    def _apply_evaluation_config(self, settings: EvaluationConfiguration):
        

    def _create_lpips_model(self) -> LPIPS:
        """Creates an instance of the :class:`lpips.LPIPS` class. Uses 'vgg' as pretrained backbone model."""
        return LPIPS(net="vgg")

    def evaluate(self, model: nn.Module, estimator: OccGridEstimator, dataset: Dataset) -> Tuple[]:
        """Evaluates the model using the provided evaluation dataset."""
        ndc = ?
        rgbs_gt = []
        rgbs_predicted = []
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
        for sample in data_loader:
            rgb_gt, pose = sample
            rgbs_gt.append(rgb_gt)
            rgb_predicted, _ = R.render_frame(
                hwf,
                data_loader.dataset.near,
                data_loader.dataset.far,
                pose[0],
                chunksize,
                estimator,
                model,
                train=False,
                ndc=ndc,
                white_bkgd=white_bkgd,
                render_step_size=render_step_size,
                device=device
            )
            rgbs_predicted.append(rgb_predicted)
        
        # Group list of tensors into a single tensor
        rgbs_gt = torch.permute(torch.stack(rgbs_gt, dim=0), (0, 3, 1, 2))
        rgbs_gt = torch.permute(torch.cat(rgbs_gt, dim=0), (0, 3, 1, 2))
        rgbs_gt = rgbs_gt.to(device)
        val_size = len(data_loader)
        psnr = self._compute_psnr_metric(rgbs_predicted, rgbs_gt)
        lpips = self._compute_lpips_metric(rgbs_predicted, rgbs_gt)
        ssim = self._compute_ssim_metric(rgbs_predicted, rgbs_gt)
        
        return psnr, lpips, ssim
    
    def _compute_psnr_metric(self, rgbs_predicted: torch.Tensor, rgbs_gt: torch.Tensor) -> :
        """Computes peak signal-to-noise ratio."""
        return -10.0 * torch.log10(nn.functional.mse_loss(rgbs_predicted, rgbs_gt))
    
    def _compute_lpips_metric(self, rgbs_predicted: torch.Tensor, rgbs_gt: torch.Tensor) -> torch.Tensor:
        """Computes the LPIPS metric."""

    def _compute_ssim_metric(self, rgbs_predicted: torch.Tensor, rgbs_gt: torch.Tensor) -> torch.Tensor:
        """Computes structure similarity measure index. Computation is performed in cpu."""
        rgbs_predicted = torch.permute(rgbs_predicted, (0, 2, 3, 1)).cpu().numpy()
        rgbs_gt = torch.permute(rgbs_gt, (0, 2, 3, 1)).cpu().numpy()

        ssim = 0.
        for rgb_predicted, rgb_gt in zip(rgb_predicted, rgbs_gt):
            ssim += SSIM(
            rgb_predicted, rgb_gt, channel_axis=-1, data_range=1.0, gaussian_weights=True
            )

        ssim /= len(rgbs_predicted)

        return ssim
