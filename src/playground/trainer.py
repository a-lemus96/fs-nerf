import os
from argparse import Namespace
from dataclasses import dataclass

import wandb
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import device as Device
from torch import nn
from torch.utils.data import Dataset

# custom modules
from render.renderer import Renderer
from core import LrScheduler
from playground.evaluator import ModelEvaluator
from utils import load_or_create_config

DEFAULT_TRAINING_CONFIG_PATH = "../configs/training.yaml"

_DEFAULTS = {
    "n_iters": 50000,
    "warmup_iters": 512,
    "warmup_mult": 0.01,
    "batch_size": 1024,
    "lr": 5e-4,
    "decay_rate": 0.1,
    "optimizer": "adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
}

@dataclass
class TrainingConfig:
    """
    Holds all hyperparameters and components required to configure a
    ModelTrainer.

    num_iterations and learning_rate may be overridden from the CLI; when
    not given (None), they fall back to the training YAML config file, as
    do the remaining fields.

    Fields:
        - num_iterations (int):             total number of training iterations
        - batch_size (int):                 number of rays per gradient step
        - learning_rate (float):            initial learning rate
        - decay_rate (float):               exponential decay rate for the
                                            learning rate scheduler
        - optimizer (str):                  name of the optimizer in use, purely
                                            descriptive — the implementation is
                                            hardcoded to torch.optim.Adam in
                                            ModelTrainer.__create_optimizer
        - betas (tuple[float, float]):      Adam optimizer beta coefficients
        - eps (float):                      Adam optimizer epsilon
        - aabb (list[float]):               axis-aligned bounding box, passed
                                            through to the renderer
    """

    num_iterations: int
    batch_size: int
    learning_rate: float
    decay_rate: float
    optimizer: str
    betas: tuple[float, float]
    eps: float
    aabb: list[float]

    def __init__(
        self,
        n_iters: int | None,
        lr: float | None,
        aabb: list[float],
        config_path: str = DEFAULT_TRAINING_CONFIG_PATH,
    ):
        """
        Builds a TrainingConfig from CLI-provided hyperparameters, the
        dataset's bounding box, and the training YAML config file.

        Args:
            n_iters (int | None): total number of training iterations;
                CLI-driven, falls back to the YAML config file if None
            lr (float | None): initial learning rate; CLI-driven, falls
                back to the YAML config file if None
            aabb (list[float]): dataset axis-aligned bounding box; dataset-dependent,
                so it isn't part of the YAML config file
            config_path (str): path to the training YAML config file,
                created with default values if it doesn't exist
        """
        cfg = load_or_create_config(config_path, _DEFAULTS)
        self.num_iterations = n_iters if n_iters is not None else cfg["n_iters"]
        self.warmup_iters = cfg["warmup_iters"]
        self.warmup_mult = cfg["warmup_mult"]
        self.batch_size = cfg["batch_size"]
        self.learning_rate = lr if lr is not None else cfg["lr"]
        self.decay_rate = cfg["decay_rate"]
        self.optimizer = cfg["optimizer"]
        self.betas = tuple(cfg["betas"])
        self.eps = cfg["eps"]
        self.aabb = aabb


class ModelTrainer:
    """
    Trainer for NeRF-like models using occupancy-grid-accelerated rendering.

    The training dataset is preloaded onto GPU at the start of fit(), and
    batches are sampled directly via torch.randint — no DataLoader or worker
    processes are used. This is efficient and correct for the small ray
    datasets typical in few-shot NeRF (a few tens of images).

    Owns its TrainingConfig (built here, from the parsed CLI args, the
    dataset's bounding box, and the training YAML config file) so callers
    deal with a single, project-level interface instead of TrainingConfig
    directly.
    """

    def __init__(
        self,
        training_device: Device,
        aabb: list[float],
        monitor_data: Dataset | None = None,
        *,
        args: Namespace,
        seed: int,
    ):
        """
        Args:
            training_device (Device): device to run training on
            aabb (list[float]): dataset axis-aligned bounding box, passed
                through to the renderer
            monitor_data (Dataset | None): single-view dataset used to
                monitor training progress
            args (Namespace): parsed command-line arguments; n_iters, lr,
                and debug are unpacked from it — n_iters/lr fall back to
                the training YAML config file when None
            seed (int): seeds the renderer's occupancy estimator's dedicated
                generator, independent of the model's own RNG stream
        """
        settings = TrainingConfig(args.n_iters, args.lr, aabb)
        self.training_device = training_device
        self.monitor_data = monitor_data
        self.configure(settings, seed, args.debug)

    def configure(self, settings: TrainingConfig, seed: int, debug: bool = False):
        """
        Applies a TrainingConfig to the trainer, setting up the renderer
        (and the occupancy grid estimator it owns) and storing all training
        hyperparameters. Called at construction and can be called again to
        reconfigure.

        Args:
            settings (TrainingConfig): full training configuration
            seed (int): seeds the renderer's occupancy estimator generator,
                as well as this trainer's own generator for batch sampling
            debug (bool): if True, disables all wandb logging
        """
        self.__apply_training_config(settings)
        self.renderer = Renderer(settings.aabb, seed)
        self.data_generator = torch.Generator(
            device=self.training_device
        ).manual_seed(seed)
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
        self.betas = settings.betas
        self.eps = settings.eps
        self.batch_size = settings.batch_size
        self.num_iterations = settings.num_iterations

    def fit(
        self,
        model: nn.Module,
        dataset: Dataset,
        evaluator: ModelEvaluator | None = None,
    ):
        """
        Runs the training loop for a given model and dataset.

        The dataset is expected to already be on the training device before fit()
        is called — use dataset.to(device) in train.py alongside val and test.
        Each dataset item holds the full set of rays for one image; at each
        iteration a random image is drawn and a random batch of pixels is
        sampled from it. At each iteration:
            1. Samples a random image, then a random batch of rays from it.
            2. Renders the batch using the current model and renderer.
            3. Computes the total loss as a sum of active loss terms.
            4. Zeroes gradients, performs a backward pass, and steps the optimizer
               and learning rate scheduler.
            5. Updates the occupancy estimator via the renderer.
            6. Optionally evaluates on monitor_data every evaluator.val_every
               iterations, if monitor_data was given at construction.
               Validation is diagnostic only — it never affects checkpoint
               selection; the model at the final iteration is what gets
               saved and evaluated.

        Logs train PSNR and learning rate to wandb at every iteration unless
        debug mode is active.

        Args:
            model (nn.Module): NeRF-like model to train
            dataset (Dataset): ray-based training dataset
            evaluator (ModelEvaluator | None): evaluator instance to use for
                validation. If None, validation is skipped. Its own
                val_every controls the cadence of validation steps;
                a val_every below 1 disables validation entirely.
        """
        if evaluator is not None:
            evaluator.set_hwf(dataset.hwf)

        self.optimizer = self.__create_optimizer(model, self.learning_rate)
        self.lr_scheduler = self.__create_lr_scheduler()

        model.to(self.training_device)
        self.renderer.to(self.training_device)

        # Dataset is expected to already be on the training device.
        # Call dataset.to(device) in train.py before fit() is called.
        n_images = len(dataset)
        n_pixels = dataset.hwf[0] * dataset.hwf[1]

        progress_bar = self.__setup_progress_bar(
            self.num_iterations, bar_description="[fit]"
        )

        run_validation = (
            evaluator is not None
            and self.monitor_data is not None
            and evaluator.val_every >= 1
        )

        model.train()
        self.renderer.train()

        for k in progress_bar:
            # Sample a random image, then a random batch of rays from it
            image_idx = torch.randint(
                0, n_images, (1,), generator=self.data_generator,
                device=self.training_device,
            ).item()
            (ray_origins, ray_dirs, rgb_gts) = dataset[image_idx]

            pixel_idxs = torch.randint(
                0, n_pixels, (self.batch_size,), generator=self.data_generator,
                device=self.training_device,
            )
            ray_origins = ray_origins[pixel_idxs]
            ray_dirs = ray_dirs[pixel_idxs]
            rgb_gts = rgb_gts[pixel_idxs]

            result = self.renderer.render_rays(ray_origins, ray_dirs, model)

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
            if run_validation and (k + 1) % evaluator.val_every == 0:
                val_psnr, val_ssim, val_lpips, val_average = evaluator.evaluate(
                    model, self.renderer, self.monitor_data
                )
                metrics.update(
                    {
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "val_lpips": val_lpips,
                        "val_average": val_average,
                    }
                )

            if not self.debug_mode:
                wandb.log(metrics)

        return

    def __training_step(
        self, current_iteration: int, model: nn.Module, loss: torch.Tensor
    ):
        """
        Performs a single gradient update step and updates the occupancy
        estimator via the renderer.

        Args:
            current_iteration (int): current training iteration index
            model (nn.Module): model being trained
            loss (torch.Tensor): scalar total loss for this iteration
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.renderer.step(current_iteration, model)

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
        optimizer = torch.optim.Adam(
            params, lr=learning_rate, betas=self.betas, eps=self.eps
        )
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
