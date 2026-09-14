import os
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import device as Device
from torch import nn
from torch.utils.data import Dataset

import render.rendering as R
from core import LrScheduler
from playground.evaluator import ModelEvaluator
from playground.estimator import OccupancyEstimator

import wandb
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """
    Holds all hyperparameters and components required to configure a
    ModelTrainer.

    Scalar hyperparameters are parsed from a command-line argparse.Namespace.

    Fields:
        - num_iterations (int):             total number of training iterations
        - batch_size (int):                 number of rays per gradient step
        - learning_rate (float):            initial learning rate
        - decay_rate (float):               exponential decay rate for the
                                            learning rate scheduler
        - aabb (List[float]):               axis-aligned bounding box, passed
                                            through to the occupancy estimator
    """

    num_iterations: int
    batch_size: int
    learning_rate: float
    decay_rate: float
    aabb: List[float]

    def __init__(
        self,
        args: Namespace,
        aabb: List[float],
    ):
        """
        Builds a TrainingConfig from a parsed argument namespace and the
        dataset's bounding box.

        Args:
            args (Namespace):                       parsed command-line arguments
            aabb (List[float]):                     dataset axis-aligned bounding box
        """
        self.num_iterations = args.n_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.lro
        self.decay_rate = args.decay_rate
        self.aabb = aabb


class ModelTrainer:
    """
    Trainer for NeRF-like models using occupancy-grid-accelerated rendering.

    The training dataset is preloaded onto GPU at the start of fit(), and
    batches are sampled directly via torch.randint — no DataLoader or worker
    processes are used. This is efficient and correct for the small ray
    datasets typical in few-shot NeRF (a few tens of images).
    """

    def __init__(
        self,
        training_device: Device,
        settings: TrainingConfig,
        monitor_data: Optional[Dataset] = None,
        debug: bool = False,
    ):
        """
        Args:
            training_device (Device): device to run training on
            settings (TrainingConfig): full training configuration
            monitor_data (Dataset | None): single-view dataset used to
                monitor training progress
            debug (bool): if True, disables all wandb logging
        """
        self.training_device = training_device
        self.monitor_data = monitor_data
        self.configure(settings, debug)

    def configure(self, settings: TrainingConfig, debug: bool = False):
        """
        Applies a TrainingConfig to the trainer, setting up the
        occupancy grid estimator and storing all training hyperparameters.
        Called at construction and can be called again to reconfigure.

        Args:
            settings (TrainingConfig): full training configuration
            debug (bool): if True, disables all wandb logging
        """
        self.__apply_training_config(settings)
        self.estimator = OccupancyEstimator(settings.aabb)
        self.render_step_size = self.estimator.render_step_size
        self.early_stop_eps = self.estimator.early_stop_eps
        self.debug_mode = debug

    def __apply_training_config(self, settings: TrainingConfig):
        """
        Unpacks scalar hyperparameters from a TrainingConfig onto the
        trainer instance.

        Args:
            settings (TrainingConfig): full training configuration
        """
        self.learning_rate = settings.learning_rate
        self.decay_rate = settings.decay_rate
        self.batch_size = settings.batch_size
        self.num_iterations = settings.num_iterations

    def fit(
        self,
        model: nn.Module,
        dataset: Dataset,
        evaluator: Optional[ModelEvaluator] = None,
        val_every: int = 500,
    ):
        """
        Runs the training loop for a given model and dataset.

        The dataset is expected to already be on the training device before fit()
        is called — use dataset.to(device) in train.py alongside val and test.
        Each dataset item holds the full set of rays for one image; at each
        iteration a random image is drawn and a random batch of pixels is
        sampled from it. At each iteration:
            1. Samples a random image, then a random batch of rays from it.
            2. Renders the batch using the current model and occupancy estimator.
            3. Computes the total loss as a sum of active loss terms.
            4. Zeroes gradients, performs a backward pass, and steps the optimizer
               and learning rate scheduler.
            5. Updates the occupancy estimator.
            6. Optionally evaluates on monitor_data every val_every iterations,
               if monitor_data was given at construction. Validation is
               diagnostic only — it never affects checkpoint selection; the
               model at the final iteration is what gets saved and evaluated.

        Logs train PSNR and learning rate to wandb at every iteration unless
        debug mode is active.

        Args:
            model (nn.Module): NeRF-like model to train
            dataset (Dataset): ray-based training dataset
            evaluator (ModelEvaluator | None): evaluator instance to use for
                validation. If None, validation is skipped.
            val_every (int): number of iterations between validation steps
        """
        self.optimizer = self.__create_optimizer(model, self.learning_rate)
        self.lr_scheduler = self.__create_lr_scheduler()

        model.to(self.training_device)
        self.estimator.to(self.training_device)

        # Dataset is expected to already be on the training device.
        # Call dataset.to(device) in train.py before fit() is called.
        n_images = len(dataset)
        n_pixels = dataset.hwf[0] * dataset.hwf[1]

        progress_bar = self.__setup_progress_bar(
            self.num_iterations, bar_description="[fit]"
        )

        run_validation = evaluator is not None and self.monitor_data is not None
        for k in progress_bar:
            model.train()
            self.estimator.train()

            # Sample a random image, then a random batch of rays from it
            image_idx = torch.randint(
                0, n_images, (1,), device=self.training_device
            ).item()
            (ray_origins, ray_dirs, rgb_gts) = dataset[image_idx]

            pixel_idxs = torch.randint(
                0, n_pixels, (self.batch_size,), device=self.training_device
            )
            ray_origins = ray_origins[pixel_idxs]
            ray_dirs = ray_dirs[pixel_idxs]
            rgb_gts = rgb_gts[pixel_idxs]

            result = R.render_rays(
                rays_o=ray_origins,
                rays_d=ray_dirs,
                estimator=self.estimator,
                model=model,
                train=True,
                render_step_size=self.render_step_size,
                early_stop_eps=self.early_stop_eps,
                device=self.training_device,
            )

            # photometric loss
            loss = F.mse_loss(result.rgb, rgb_gts)
            with torch.no_grad():
                psnr = -10.0 * torch.log10(loss).item()

            metrics = {
                "train_psnr": psnr,
                "photo_loss": loss.item(),
                "lr": self.lr_scheduler.lr,
            }

            self.__training_step(k, model, loss)

            # periodic validation
            if run_validation and (k + 1) % val_every == 0:
                model.eval()
                self.estimator.eval()
                val_psnr, val_ssim, val_lpips, val_average = evaluator.evaluate(
                    model, self.estimator, self.monitor_data
                )
                metrics.update(
                    {
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_lpips": val_lpips,
                        "val_average": val_average,
                    }
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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.estimator.step(current_iteration, model)

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

    def __create_lr_scheduler(self) -> LrScheduler:
        """
        Instantiates the exponential-decay learning rate scheduler.

        Returns:
            LrScheduler: configured learning rate scheduler
        """
        return LrScheduler(
            self.optimizer, self.num_iterations, self.learning_rate, self.decay_rate
        )
