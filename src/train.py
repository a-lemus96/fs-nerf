# stdlib imports
import json
import logging
import os
import random

# third-party imports
import numpy as np
import torch
from torch import nn
import wandb

# local imports
from core import (
    FrequencyRegularizer,
    ConstantScheduler,
    LinearScheduler,
    Nerf,
    Sinerf,
    WeightSumSquaredRegularizer,
)
from llff import LLFFDataset
import utils.parser as P
from utils import create_split_file, load_split
from playground import ModelTrainer, TrainingConfiguration, ModelEvaluator, EvaluationConfiguration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RANDOM SEED
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = P.config_parser()  # parse command line arguments


def main():
    device = get_computing_device()
    if not args.debug:
        run = init_wandb()

    if not os.path.isfile("../configs/split.yaml"):
        create_split_file()
        logger.info("Split file created at ../configs/split.yaml")

    train_ids, eval_ids, monitor_id = load_split(scene=args.scene, n_imgs=args.n_imgs)
    train_data = LLFFDataset(scene=args.scene, img_ids=train_ids)
    eval_data = LLFFDataset(scene=args.scene, img_ids=eval_ids)
    monitor_data = LLFFDataset(scene=args.scene, img_ids=[monitor_id])

    train_data.to(device)
    eval_data.to(device)
    monitor_data.to(device)

    # Resolve output directory: flat structure out_dir/<run_id>/
    if not args.debug:
        out_dir = os.path.normpath(os.path.join(args.out_dir, run.id))
        os.makedirs(out_dir, exist_ok=True)

    if not args.render_only:
        model = init_model()

        # build training settings
        training_settings = TrainingConfiguration(device, args)
        # build frequency regularizer
        if args.alpha is not None:
            match args.freq_scheduler:
                case "constant":
                    scheduler = ConstantScheduler(alpha=args.alpha)
                case "linear":
                    T = int(args.reg_ratio * args.n_iters)
                    scheduler = LinearScheduler(alpha_0=args.alpha, alpha_T=0.0, T=T)
            freq_regularizer = FrequencyRegularizer(
                model.named_parameters(),
                scheduler,
                reg=args.reg,
            )
        else:
            freq_regularizer = None     
        occl_regularizer = WeightSumSquaredRegularizer() if args.beta is not None else None
        training_settings.occl_regularizer = occl_regularizer
        training_settings.freq_regularizer = freq_regularizer
        # TODO: Temporary workaround but probably need to move OccGridConfig one level up
        training_settings.occupancy_estimator_settings.aabb = train_data.aabb

        model_trainer = ModelTrainer(training_settings, monitor_data, args.debug)

        eval_settings = EvaluationConfiguration(device, train_data.hwf, args)
        model_evaluator = ModelEvaluator(eval_settings, debug=args.debug)

        # trains model using the trainer's configuration
        model_trainer.fit(
            model,
            train_data,
            evaluator=model_evaluator,
            out_dir=out_dir if not args.debug else None,
        )

        # final evaluation on test set
        estimator = model_trainer.estimator
        model.eval()
        estimator.eval()
        final_psnr, final_ssim, final_lpips, final_average = model_evaluator.evaluate(
            model, estimator, eval_data
        )

        if not args.debug:
            metrics = {
                "final_psnr": final_psnr,
                "final_ssim": final_ssim,
                "final_lpips": final_lpips,
                "final_average": final_average,
            }
            wandb.log(metrics)

            # Save metrics as JSON (model is saved during training via trainer)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # Save config (args) as JSON
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            # Log best model checkpoint to wandb
            wandb.log_model(os.path.join(out_dir, "best_model.pt"))


def get_computing_device() -> torch.device:
    computing_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing device: {torch.cuda.get_device_name(computing_device)}")
    return computing_device


def init_wandb():
    wandb.login()
    # set up wandb run to track training
    alpha_str = f"a{args.alpha}" if args.alpha else "a0"
    beta_str = f"b{args.beta}" if args.beta else "b0"
    name = (
        f"{args.model}_{args.scene}_nimg{args.n_imgs}_{alpha_str}_{beta_str}_{args.reg}"
    )
    run = wandb.init(project="fs-nerf", name=name, config=args)
    return run


def init_model() -> nn.Module:
    """
    Initialize NeRF-like model.
    ----------------------------------------------------------------------------
    Args:
        None
    Returns:
        nn.Module: model
    """
    # instantiate model
    match args.model:
        case "nerf":
            model = Nerf()
        case "sinerf":
            model = Sinerf()
        case _:
            raise ValueError(f"Model {args.model} not supported")

    return model


if __name__ == "__main__":
    main()
