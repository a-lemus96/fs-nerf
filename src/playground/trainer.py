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
from playground.renderer import Renderer
from core import LrScheduler, ConstantLrScheduler, ExponentialLrScheduler
from playground.evaluator import ModelEvaluator
from utils import load_or_create_config

DEFAULT_TRAINING_CONFIG_PATH = "../configs/training.yaml"

_DEFAULTS = {
    "n_iters": 50000,
    "warmup_iters": 512,
    "warmup_mult": 0.01,
    "batch_size": 1024,
    "lr": {"sinerf": 0.001, "nerf": 0.0005},
    "decay_rate": 0.1,
    "lr_scheduler_type": "exp",
    "optimizer": "adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
}

_LR_SCHEDULERS = ("exp", "constant")


@dataclass
class TrainingConfig:
    """
    Holds all hyperparameters and components required to configure a
    ModelTrainer.

    num_iterations and lr_scheduler_type may be overridden from the CLI;
    when not given (None), they fall back to the training YAML config file,
    as do the remaining fields. The learning rate is stored as per-model
    defaults (lr_values); ModelTrainer selects the entry for the chosen
    model, unless --lr overrides it.

    Fields:
        - num_iterations (int):             total number of training iterations
        - batch_size (int):                 number of rays per gradient step
        - lr_values (dict[str, float]):     default initial learning rate values
        - decay_rate (float):               exponential decay rate for the
                                            learning rate scheduler
        - lr_scheduler_type (str):          learning rate scheduler type, one
                                            of _LR_SCHEDULERS ("exp" or
                                            "constant")
        - optimizer (str):                  name of the optimizer in use, purely
                                            descriptive — the implementation is
                                            hardcoded to torch.optim.Adam in
                                            ModelTrainer.__create_optimizer
        - betas (tuple[float, float]):      Adam optimizer beta coefficients
        - eps (float):                      Adam optimizer epsilon
    """

    num_iterations: int
    batch_size: int
    lr_values: dict[str, float]
    decay_rate: float
    lr_scheduler_type: str
    optimizer: str
    betas: tuple[float, float]
    eps: float

    def __init__(
        self,
        args: Namespace,
        config_path: str = DEFAULT_TRAINING_CONFIG_PATH,
    ):
        """
        Builds a TrainingConfig from the parsed CLI arguments and the
        training YAML config file.

        Args:
            args (Namespace): parsed command-line arguments; n_iters and
                lr_scheduler_type are unpacked from it and fall back
                to the YAML config file when None (lr is resolved per
                model in ModelTrainer)
            config_path (str): path to the training YAML config file,
                created with default values if it doesn't exist
        """
        n_iters, lr_scheduler_type = (args.n_iters, args.lr_scheduler_type)
        cfg = load_or_create_config(config_path, _DEFAULTS)
        self.num_iterations = n_iters if n_iters is not None else cfg["n_iters"]
        self.warmup_iters = cfg["warmup_iters"]
        self.warmup_mult = cfg["warmup_mult"]
        self.batch_size = cfg["batch_size"]
        self.lr_values = cfg["lr"]
        self.decay_rate = cfg["decay_rate"]
        self.lr_scheduler_type = (
            lr_scheduler_type
            if lr_scheduler_type is not None
            else cfg.get("lr_scheduler_type", _DEFAULTS["lr_scheduler_type"])
        )
        if self.lr_scheduler_type not in _LR_SCHEDULERS:
            raise ValueError(
                f"Unknown lr_scheduler_type '{self.lr_scheduler_type}'; "
                f"expected one of {_LR_SCHEDULERS}."
            )
        self.optimizer = cfg["optimizer"]
        self.betas = tuple(cfg["betas"])
        self.eps = cfg["eps"]


class ModelTrainer:
    """
    Trainer for NeRF-like models using occupancy-grid-accelerated rendering.

    The training dataset is preloaded onto GPU at the start of fit(), and
    batches are sampled directly via torch.randint — no DataLoader or worker
    processes are used. This is efficient and correct for the small ray
    datasets typical in few-shot NeRF (a few tens of images).

    Owns its TrainingConfig (built here, from the parsed CLI args and the
    training YAML config file) so callers deal with a single, project-level
    interface instead of TrainingConfig directly.
    """

    def __init__(
        self,
        training_device: Device,
        monitor_data: Dataset | None = None,
        *,
        args: Namespace,
        seed: int,
    ):
        """
        Args:
            training_device (Device): device to run training on
            monitor_data (Dataset | None): single-view dataset used to
                monitor training progress
            args (Namespace): parsed command-line arguments; n_iters, lr,
                lr_scheduler_type, and debug are unpacked from it —
                n_iters/lr/lr_scheduler_type fall back to the training
                YAML config file when None
            seed (int): seeds the renderer's occupancy estimator's dedicated
                generator, independent of the model's own RNG stream
        """
        settings = TrainingConfig(args)
        self.training_device = training_device
        self.monitor_data = monitor_data
        self.configure(settings, seed, args)
        

    def configure(self, settings: TrainingConfig, seed: int, args: Namespace):
        """
        Applies a TrainingConfig to the trainer, setting up the renderer
        (and the occupancy grid estimator it owns) and storing all training
        hyperparameters. Called at construction and can be called again to
        reconfigure.

        Args:
            settings (TrainingConfig): full training configuration
            seed (int): seeds the renderer's occupancy estimator generator,
                as well as this trainer's own generator for batch sampling
            args (Namespace): parsed command-line arguments; model and lr
                select the initial learning rate, and debug, if True,
                disables all wandb logging
        """
        self.__apply_training_config(settings, args)
        self.renderer = Renderer(seed)
        self.data_generator = torch.Generator(
            device=self.training_device
        ).manual_seed(seed)
        self.debug_mode = args.debug

    def __apply_training_config(self, settings: TrainingConfig, args: Namespace):
        """
        Unpacks scalar hyperparameters from a TrainingConfig onto the
        trainer instance.

        Args:
            settings (TrainingConfig): full training configuration
            args (Namespace): parsed command-line arguments; model and lr
                determine the initial learning rate
        """
        if args.lr is not None:
            self.learning_rate = args.lr
        else:
            try:
                self.learning_rate = settings.lr_values[args.model]
            except KeyError:
                raise ValueError(f"Model type '{args.model}' has no default lr value.") from None
        self.decay_rate = settings.decay_rate
        self.lr_scheduler_type = settings.lr_scheduler_type
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

        Logs the photometric loss and learning rate to wandb at every
        iteration unless debug mode is active. Validation metrics and images
        are logged by the evaluator itself, into the same wandb step.

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

            metrics = {
                "photo_loss": loss.item(),
                "lr": self.lr_scheduler.lr,
            }

            self.__training_step(k, model, loss)

            # periodic validation
            if run_validation and (k + 1) % evaluator.val_every == 0:
                # logs its own metrics/images into this iteration's wandb step
                evaluator.evaluate(model, self.renderer, self.monitor_data)

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
        Instantiates the learning rate scheduler selected by
        lr_scheduler_type ("exp" or "constant").

        Returns:
            LrScheduler: configured learning rate scheduler
        """
        if self.lr_scheduler_type == "exp":
            return ExponentialLrScheduler(
                self.optimizer,
                self.num_iterations,
                self.learning_rate,
                self.decay_rate,
            )
        if self.lr_scheduler_type == "constant":
            return ConstantLrScheduler(self.optimizer, self.learning_rate)
        raise ValueError(f"Unknown lr_scheduler_type '{self.lr_scheduler_type}'.")
