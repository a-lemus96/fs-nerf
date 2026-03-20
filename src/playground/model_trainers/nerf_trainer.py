import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

import render.rendering as R
from core.lr_scheduler import ExponentialDecay, Constant
from core.freq_regularizer import FrequencyRegularizer
from core.occlusion import OcclusionRegularizer
from nerfacc.estimators.occ_grid import OccGridEstimator
from playground.model_evaluators.model_evaluator_base import ModelEvaluatorBase
from playground.training_configuration import TrainingConfiguration
from playground.occ_estimator_configuration import OccupancyGridEstimatorConfiguration

import wandb
from tqdm import tqdm


class NeRFModelTrainer:
    """
    Trainer for NeRF-like models using occupancy-grid-accelerated rendering.

    The training dataset is preloaded onto GPU at the start of fit(), and
    batches are sampled directly via torch.randint — no DataLoader or worker
    processes are used. This is efficient and correct for the small ray
    datasets typical in few-shot NeRF (a few tens of images).
    """

    def __init__(self, settings: TrainingConfiguration, debug: bool = False):
        """
        Args:
            settings (TrainingConfiguration): full training configuration
            debug (bool): if True, disables all wandb logging
        """
        self.best_val_psnr = float("-inf")
        self.configure(settings, debug)

    def configure(self, settings: TrainingConfiguration, debug: bool = False):
        """
        Applies a TrainingConfiguration to the trainer, setting up the
        occupancy grid estimator and storing all training hyperparameters.
        Called at construction and can be called again to reconfigure.

        Args:
            settings (TrainingConfiguration): full training configuration
            debug (bool): if True, disables all wandb logging
        """
        self.__apply_training_config(settings)
        estimator_settings = settings.occupancy_estimator_settings
        self.render_step_size = estimator_settings.render_step_size
        self.estimator = self.__create_occupancy_estimator(estimator_settings)
        self.debug_mode = debug
        self.occl_regularizer: Optional[OcclusionRegularizer] = (
            settings.occl_regularizer
        )
        self.occl_beta: Optional[float] = settings.occl_beta

    def __apply_training_config(self, settings: TrainingConfiguration):
        """
        Unpacks scalar hyperparameters from a TrainingConfiguration onto the
        trainer instance.

        Args:
            settings (TrainingConfiguration): full training configuration
        """
        self.training_device = settings.training_device
        self.learning_rate = settings.learning_rate
        self.lr_scheduler_type = settings.lr_scheduler_type
        self.lr_scheduler_kwargs = settings.lr_scheduler_kwargs
        self.batch_size = settings.batch_size
        self.num_iterations = settings.num_iterations
        self.weight_decay_importance = settings.weight_decay_importance
        self.weight_decay_reg_fn = settings.weight_decay_reg_fn
        self.white_background = settings.white_background
        self.freq_regularizer: Optional[FrequencyRegularizer] = (
            settings.freq_regularizer
        )

    def fit(
        self,
        model: nn.Module,
        dataset: Dataset,
        evaluator: Optional[ModelEvaluatorBase] = None,
        val_dataset: Optional[Dataset] = None,
        val_every: int = 500,
        out_dir: Optional[str] = None,
    ):
        """
        Runs the training loop for a given model and dataset.

        The dataset is expected to already be on the training device before fit()
        is called — use dataset.to(device) in run-nerf.py alongside val and test.
        At each iteration:
            1. Samples a batch of rays randomly via torch.randint.
            2. Renders the batch using the current model and occupancy estimator.
            3. Computes the total loss as a sum of active loss terms.
            4. Zeroes gradients, performs a backward pass, and steps the optimizer
               and learning rate scheduler.
            5. Updates the occupancy estimator.
            6. Optionally evaluates on val_dataset every val_every iterations.
                6.1. Optionally saves the model with highest validation PSNR.

        Logs train PSNR, learning rate, frequency regularization weight, and
        occlusion loss to wandb at every iteration unless debug mode is active.

        Args:
            model (nn.Module): NeRF-like model to train
            dataset (Dataset): ray-based training dataset
            evaluator (ModelEvaluatorBase | None): evaluator instance to use for
                validation. If None, validation is skipped.
            val_dataset (Dataset | None): validation dataset. If None, validation
                is skipped even if an evaluator is provided.
            val_every (int): number of iterations between validation steps
        """
        self.optimizer = self.__create_optimizer(model, self.learning_rate)
        self.lr_scheduler = self.__create_lr_scheduler(
            self.lr_scheduler_type, **self.lr_scheduler_kwargs
        )

        model.to(self.training_device)
        self.estimator.to(self.training_device)

        # Dataset is expected to already be on the training device.
        # Call dataset.to(device) in run-nerf.py before fit() is called.
        n_rays = len(dataset)

        alpha = self.weight_decay_importance

        progress_bar = self.__setup_progress_bar(
            self.num_iterations, bar_description="[fit]"
        )

        run_validation = evaluator is not None and val_dataset is not None
        self.best_val_psnr = float("-inf")
        for k in progress_bar:
            model.train()
            self.estimator.train()

            # Sample a random batch of rays directly from GPU tensors
            idxs = torch.randint(
                0, n_rays, (self.batch_size,), device=self.training_device
            )
            ray_origins = dataset.rays_o[idxs]
            ray_dirs = dataset.rays_d[idxs]
            rgb_ground_truths = dataset.rgb[idxs]

            result = R.render_rays(
                rays_o=ray_origins,
                rays_d=ray_dirs,
                estimator=self.estimator,
                model=model,
                train=True,
                white_bkgd=self.white_background,
                render_step_size=self.render_step_size,
                device=self.training_device,
            )

            # photometric loss
            loss = F.mse_loss(result.rgb, rgb_ground_truths)
            with torch.no_grad():
                psnr = -10.0 * torch.log10(loss).item()

            # frequency regularization
            if self.freq_regularizer is not None:
                loss += self.freq_regularizer()

            # occlusion regularization
            metrics = {
                "train_psnr": psnr,
                "photo_loss": loss.item(),
                "lr": self.lr_scheduler.lr,
                "alpha": (
                    self.freq_regularizer.freq_scheduler.alpha
                    if self.freq_regularizer is not None
                    else None
                ),
            }
            if (
                self.occl_regularizer is not None
                and result.weights is not None
                and result.weights.numel() > 0
            ):
                occl_loss = self.occl_beta * self.occl_regularizer(result)
                loss += occl_loss
                if not self.debug_mode:
                    metrics["occl_loss"] = occl_loss.item()

            self.__training_step(k, model, loss)

            # periodic validation
            if run_validation and (k + 1) % val_every == 0:
                model.eval()
                self.estimator.eval()
                val_psnr, val_ssim, val_lpips = evaluator.evaluate(
                    model, self.estimator, val_dataset
                )
                metrics.update(
                    {
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_lpips": val_lpips,
                    }
                )
                # save best model
                if out_dir is not None and val_psnr > self.best_val_psnr:
                    self.best_val_psnr = val_psnr
                    torch.save(
                        model.state_dict(), os.path.join(out_dir, "best_model.pt")
                    )
                model.train()
                self.estimator.train()

            if not self.debug_mode:
                wandb.log(metrics)

        return

    def __training_step(
        self, current_iteration: int, model: nn.Module, loss: torch.Tensor
    ):
        """
        Performs a single gradient update step and updates the occupancy
        estimator.

        Args:
            current_iteration (int): current training iteration index
            model (nn.Module): model being trained
            loss (torch.Tensor): scalar total loss for this iteration
        """
        if self.freq_regularizer is not None:
            self.freq_regularizer.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.__update_occupancy_estimator(current_iteration, model)

    def __update_occupancy_estimator(self, current_iteration: int, model: nn.Module):
        """
        Steps the occupancy grid estimator using the current model's density
        predictions. Called at every training iteration.
        ------------------------------------------------------------------------
        Args:
            current_iteration (int): current training iteration index
            model (nn.Module): model used to evaluate occupancy
        """

        def occ_eval_fn(x):
            return model(x) * self.render_step_size

        with torch.cuda.amp.autocast():
            self.estimator.update_every_n_steps(
                step=current_iteration, occ_eval_fn=occ_eval_fn, occ_thre=1e-2
            )

    def __create_occupancy_estimator(
        self, settings: OccupancyGridEstimatorConfiguration
    ) -> OccGridEstimator:
        """
        Instantiates an OccGridEstimator from a configuration object.

        Args:
            settings (OccupancyGridEstimatorConfiguration): estimator config
        Returns:
            OccGridEstimator: initialised occupancy grid estimator
        """
        aabb = settings.aabb
        grid_resolution = settings.grid_resolution
        grid_number_of_levels = settings.grid_num_levels
        estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_number_of_levels
        )
        return estimator

    def __setup_progress_bar(self, num_iterations: int, bar_description: str):
        """
        Creates a tqdm progress bar for the training loop.

        Args:
            num_iterations (int): total number of training iterations
            bar_description (str): label displayed on the progress bar
        Returns:
            tqdm: progress bar iterable over range(num_iterations)
        """
        return tqdm(range(num_iterations), desc=bar_description)

    def __create_optimizer(
        self, model: nn.Module, learning_rate: float
    ) -> torch.optim.Adam:
        """
        Instantiates an Adam optimizer over all model parameters.

        Args:
            model (nn.Module): model whose parameters will be optimized
            learning_rate (float): initial learning rate
        Returns:
            torch.optim.Adam: configured optimizer
        """
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        return optimizer

    def __create_lr_scheduler(self, lr_scheduler_type: str, **kwargs: Dict[str, Any]):
        """
        Instantiates a learning rate scheduler based on the specified type.

        Args:
            lr_scheduler_type (str): one of 'const' or 'exp'
            **kwargs: additional keyword arguments forwarded to the scheduler
        Returns:
            Scheduler: configured learning rate scheduler
        Raises:
            ValueError: if lr_scheduler_type is not a supported scheduler type
        """
        match (lr_scheduler_type):
            case "const":
                scheduler = Constant(
                    self.optimizer, self.num_iterations, self.learning_rate, **kwargs
                )
            case "exp":
                scheduler = ExponentialDecay(
                    self.optimizer, self.num_iterations, self.learning_rate, **kwargs
                )
            case _:
                raise ValueError(
                    f"'{lr_scheduler_type}' is not a supported lr scheduler type."
                )

        return scheduler
