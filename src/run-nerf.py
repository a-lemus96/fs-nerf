# stdlib imports
import json
import os
import random
from typing import Tuple

# third-party imports
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import wandb

# local imports
from core.freq_regularizer import FrequencyRegularizer, ConstantScheduler, LinearScheduler
import core.models as M
from nerfdata.datasets import llff, blender
from nerfdata.utils.splitter import Splitter
import render.rendering as R
import utils.parser as P
from utils.camera3dplotter import Camera3DPlotter
from playground.model_trainers.nerf_trainer import NeRFModelTrainer
from playground.model_evaluators.nerf_evaluator import NeRFModelEvaluator
from playground.training_configuration import TrainingConfiguration
from playground.configuration.evaluation_configuration import EvaluationConfiguration
from core.occlusion import VarianceRegularizer

# GLOBAL VARIABLES
k = 0  # global step counter
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000  # memory snapshot

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

    # set up dataset configuration
    dataset_config = {
        "synthetic": (blender.BlenderDataset, {"white_bkgd": args.white_bkgd}),
        "llff": (llff.LLFFDataset, {"white_bkgd": args.white_bkgd, "ndc": True}),
    }
    _, dataset_kwargs = dataset_config[args.dataset]

    # get training, validation and test datasets
    splitter = Splitter(args.dataset, args.scene, n_training_views=args.n_imgs)
    splitter.split()
    datasets = splitter.get_datasets(train_img_mode=False, **dataset_kwargs)
    train_dataset, val_dataset, test_dataset = datasets

    # camera plotter needs poses on CPU — must be called before to(device)
    if not args.debug:
        cam_plotter = create_camera_plotter(datasets)
        cam_plotter.upload_plot()

    # move all datasets to device once — avoids per-batch CPU-to-GPU transfers
    train_dataset.to(device)
    val_dataset.to(device)
    test_dataset.to(device)

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
        occl_regularizer = VarianceRegularizer() if args.beta is not None else None
        training_settings.occl_regularizer = occl_regularizer
        training_settings.freq_regularizer = freq_regularizer
        # TODO: Temporary workaround but probably need to move OccGridConfig one level up
        training_settings.occupancy_estimator_settings.aabb = train_dataset.aabb

        model_trainer = NeRFModelTrainer(training_settings, args.debug)

        eval_settings = EvaluationConfiguration(device, train_dataset.hwf, args)
        model_evaluator = NeRFModelEvaluator(eval_settings, debug=args.debug)

        # trains model using the trainer's configuration
        model_trainer.fit(
            model,
            train_dataset,
            evaluator=model_evaluator,
            val_dataset=val_dataset,
            val_every=args.val_rate,
            out_dir=out_dir if not args.debug else None,
        )

        # final evaluation on test set
        estimator = model_trainer.estimator
        model.eval()
        estimator.eval()
        final_psnr, final_ssim, final_lpips = model_evaluator.evaluate(
            model, estimator, test_dataset
        )

        if not args.debug:
            metrics = {
                "final_psnr": final_psnr,
                "final_ssim": final_ssim,
                "final_lpips": final_lpips,
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

    else:
        model = init_model()
        # load model from flat run directory
        model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))

    # compute path poses for video output
    path_poses = splitter.path_poses

    # render frames for poses
    model.eval()
    estimator = model_trainer.estimator
    estimator.eval()
    output = R.render_path(
        torch.from_numpy(path_poses).float(),
        train_dataset.hwf,
        train_dataset.near,
        train_dataset.far,
        2 * args.batch_size,
        model,
        estimator,
        ndc=train_dataset.ndc,
        white_bkgd=args.white_bkgd,
        device=device,
    )
    frames, d_frames = output

    if not args.debug:
        # put together frames and save result into .mp4 file
        frames, d_frames = R.render_video(frames=frames, d_frames=d_frames)
        # log final video renderings to wandb
        wandb.log(
            {
                "rgb_video": wandb.Video(frames, format="mp4", fps=15),
                "depth_video": wandb.Video(d_frames, format="mp4", fps=15),
            }
        )


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


def create_camera_plotter(
    datasets: Tuple[Dataset, Dataset, Dataset],
) -> Camera3DPlotter:
    train_dataset, val_dataset, test_dataset = datasets
    cam_plotter = Camera3DPlotter()

    cam_plotter.set_poses(train_dataset.poses, "train")
    cam_plotter.set_poses(val_dataset.poses, "val")
    cam_plotter.set_poses(test_dataset.poses, "test")

    cam_plotter.configure_pose_markers("train", size=7, opacity=0.8, color="black")
    cam_plotter.configure_pose_markers("val", size=7, opacity=0.8, color="red")
    cam_plotter.configure_pose_markers("test", size=7, opacity=0.8, color="blue")

    cam_plotter.set_axes_margins(left=20, right=20, top=20, bottom=20)
    # set fixed axis scales
    t = 1 if args.dataset == "llff" else 5
    factor = 1 if args.dataset == "llff" else 0
    cam_plotter.set_axes_ranges(xrange=[-t, t], yrange=[-t, t], zrange=[-t * factor, t])

    return cam_plotter


def init_model() -> nn.Module:
    """
    Initialize NeRF-like model.
    ----------------------------------------------------------------------------
    Args:
        None
    Returns:
        nn.Module: model
    """
    # keyword args for positional encoding
    kwargs = {
        "pos_fn": {"n_freqs": args.n_freqs, "log_space": args.log_space},
        "dir_fn": {"n_freqs": args.n_freqs_views, "log_space": args.log_space},
    }
    alpha_values = [30] + [1] * (args.n_layers - 1)
    # instantiate model
    match args.model:
        case "nerf":
            model = M.NeRF(
                args.d_input,
                args.d_input,
                args.n_layers,
                args.d_filter,
                args.skip,
                **kwargs,
            )
        case "sinerf":
            model = M.SiNeRF(
                args.d_input,
                args.d_input,
                args.d_filter,
                alpha_values,
            )
        case _:
            raise ValueError(f"Model {args.model} not supported")

    return model


if __name__ == "__main__":
    main()
