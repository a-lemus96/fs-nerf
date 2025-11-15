from nerfacc.estimators.occ_grid import OccGridEstimator
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from typing import Dict, Any
import wandb

from model_trainer_base import ModelTrainerBase
from training_configuration import TrainingConfiguration
from occ_estimator_configuration import OccupancyGridEstimatorConfiguration
from core.scheduler import Constant, ExponentialDecay

import render.rendering as R


class NeRFModelTrainer(ModelTrainerBase):
    """Trains a NeRF model on a specific dataset. Uses an occupancy grid estimator."""

    def __init__(self, settings: TrainingConfiguration, debug: bool = False):
        self.configure(settings, debug)

    def configure(self, settings: TrainingConfiguration, debug: bool = False):
        self.__apply_training_config(settings)
        estimator_settings = settings.occupancy_estimator_settings
        self.render_step_size = estimator_settings.render_step_size
        self.estimator = self.__create_occupancy_estimator(estimator_settings)
        self.debug_mode = debug

    def __apply_training_config(self, settings: TrainingConfiguration):
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
        self.optimizer = self.__create_optimizer(model, self.learning_rate)
        self.lr_scheduler = self.__create_lr_scheduler(
            self.optimizer, self.lr_scheduler_type, self.lr_scheduler_kwargs
        )

        model.to(self.training_device)
        self.estimator.to(self.training_device)

        alpha = self.weight_decay_importance

        progress_bar = self.__setup_progress_bar(
            self.num_iterations, bar_description=f"[fit]"
        )
        train_dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )
        iterator = iter(train_dataloader)  # data iterator

        for k in progress_bar:  # loop over the number of iterations
            model.train()
            self.estimator.train()
            # get next batch of data
            try:
                ray_origins, ray_dirs, rgb_ground_truths = next(iterator)
            except StopIteration:
                iterator = iter(train_dataloader)
                ray_origins, ray_dirs, rgb_ground_truths = next(iterator)

            (rgb_predicted, *_), _, _ = R.render_rays(
                rays_o=ray_origins,
                rays_d=ray_dirs,
                estimator=self.estimator,
                model=model,
                train=True,
                white_bkgd=self.white_background,
                render_step_size=self.render_step_size,
                device=self.training_device,
            )

            loss, psnr = self.__compute_total_loss(rgb_predicted, rgb_ground_truths)
            self.__training_step(loss)

            # log metrics
            if not self.debug_mode:
                wandb.log(
                    {"train_psnr": psnr, "lr": self.lr_scheduler.lr, "alpha": alpha}
                )

            # TODO: Compute validation
        return

    def __compute_total_loss(
        self,
        model: nn.Module,
        rgb_predicted: torch.Tensor,
        rgb_ground_truths: torch.Tensor,
        alpha: float,
    ):
        # compute loss and PSNR
        rgb_ground_truths = rgb_ground_truths.to(self.training_device)
        loss = F.mse_loss(rgb_predicted, rgb_ground_truths)
        with torch.no_grad():
            psnr = -10.0 * torch.log10(loss).item()

        # weight decay regularization
        if alpha is not None:
            freq_reg = torch.tensor(0.0).to(self.training_device)
            # linear decay schedule
            if True:
                for name, param in model.named_parameters():
                    if "weight" in name and param.shape[0] > 3:
                        if self.weight_decay_reg_fn == "l1":
                            freq_reg += torch.abs(param).sum()
                        else:
                            freq_reg += torch.square(param).sum().sqrt()

                loss += alpha * freq_reg

        return loss, psnr

    def __training_step(
        self, current_iteration: int, model: nn.Module, loss: torch.Tensor
    ):
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.__update_occupancy_estimator(current_iteration, model)

    def __update_occupancy_estimator(self, current_iteration, model):
        # define occupancy evaluation function
        def occ_eval_fn(x):
            return model(x) * self.render_step_size

        self.__update_occupancy_estimator(current_iteration, model)
        # update based on the current iteration
        with torch.cuda.amp.autocast():
            self.estimator.update_every_n_steps(
                step=current_iteration, occ_eval_fn=occ_eval_fn, occ_thre=1e-2
            )

    def __create_occupancy_estimator(
        self, settings: OccupancyGridEstimatorConfiguration
    ):
        aabb = settings.aabb
        grid_resolution = settings.grid_resolution
        grid_number_of_levels = settings.grid_num_levels
        estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_number_of_levels
        )

        return estimator

    def __setup_progress_bar(self, num_iterations: int, bar_description: str):
        return tqdm(range(num_iterations), desc=bar_description)  # set up progress bar

    def __create_optimizer(self, model: nn.Module, learning_rate: float):
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        return optimizer

    def __create_lr_scheduler(self, lr_scheduler_type: str, **kwargs: Dict[str, Any]):
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
