from nerfacc.estimators.occ_grid import OccGridEstimator
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional
import wandb

from playground.model_trainers.model_trainer_base import ModelTrainerBase
from playground.training_configuration import TrainingConfiguration
from playground.occ_estimator_configuration import OccupancyGridEstimatorConfiguration
from core.scheduler import Constant, ExponentialDecay
from core.occlusion import OcclusionRegularizer

import render.rendering as R


class NeRFModelTrainer(ModelTrainerBase):
    """
    Trains a NeRF-like model on a ray-based dataset using an occupancy grid
    estimator to accelerate sampling.

    The total loss is a sum of up to three terms, the last two of which are optional:
        - Photometric loss: MSE between rendered and ground-truth RGB values.
        - Frequency regularization: norm penalty on model weights, biasing the
          model toward low-frequency solutions during early training.
        - Occlusion regularization: penalty on the rendering weight distribution
          along each ray, encouraging thin and solid geometry.

    The trainer is decoupled from any specific occlusion regularizer
    implementation. The concrete regularizer is injected via
    TrainingConfiguration and accessed only through the OcclusionRegularizer
    abstract interface.
    """

    def __init__(self, settings: TrainingConfiguration, debug: bool = False):
        """
        Initializes the trainer by delegating to configure().

        Args:
            settings (TrainingConfiguration): full training configuration
            debug (bool): if True, disables all wandb logging
        """
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
        self.occl_regularizer: Optional[OcclusionRegularizer] = settings.occl_regularizer
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

    def fit(self, model: nn.Module, dataset: Dataset):
        """
        Runs the training loop for a given model and dataset.

        At each iteration:
            1. Samples a batch of rays from the dataset.
            2. Renders the batch using the current model and occupancy estimator.
            3. Computes the total loss as a sum of active loss terms.
            4. Performs a gradient update and steps the learning rate scheduler.
            5. Updates the occupancy estimator.

        Logs train PSNR, learning rate, frequency regularization weight, and
        occlusion loss to wandb at every iteration unless debug mode is active.

        Args:
            model (nn.Module): NeRF-like model to train
            dataset (Dataset): ray-based training dataset
        """
        self.optimizer = self.__create_optimizer(model, self.learning_rate)
        self.lr_scheduler = self.__create_lr_scheduler(
            self.lr_scheduler_type, **self.lr_scheduler_kwargs
        )

        model.to(self.training_device)
        self.estimator.to(self.training_device)

        alpha = self.weight_decay_importance

        progress_bar = self.__setup_progress_bar(
            self.num_iterations, bar_description="[fit]"
        )
        train_dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
        iterator = iter(train_dataloader)

        for k in progress_bar:
            model.train()
            self.estimator.train()

            try:
                ray_origins, ray_dirs, rgb_ground_truths = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                ray_origins, ray_dirs, rgb_ground_truths = next(iterator)

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
            rgb_ground_truths = rgb_ground_truths.to(self.training_device)
            loss = F.mse_loss(result.rgb, rgb_ground_truths)
            with torch.no_grad():
                psnr = -10.0 * torch.log10(loss).item()

            # frequency regularization
            if alpha is not None:
                freq_reg = torch.tensor(0.0).to(self.training_device)
                for name, param in model.named_parameters():
                    if "weight" in name and param.shape[0] > 3:
                        if self.weight_decay_reg_fn == "l1":
                            freq_reg += torch.abs(param).sum()
                        else:
                            freq_reg += torch.square(param).sum()
                loss += alpha * freq_reg

            # occlusion regularization
            metrics = {"train_psnr": psnr, "lr": self.lr_scheduler.lr, "alpha": alpha}
            if self.occl_regularizer is not None and result.weights is not None and result.weights.numel() > 0:
                occl_loss = self.occl_regularizer(result)
                loss += self.occl_beta * occl_loss
                if not self.debug_mode:
                    metrics["occl_loss"] = occl_loss.item()

            self.__training_step(k, model, loss)

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
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
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

    def __create_optimizer(self, model: nn.Module, learning_rate: float) -> torch.optim.Adam:
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
