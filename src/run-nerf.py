# stdlib imports
import logging
import os
import random
from typing import List, Tuple

# third-party imports
from lpips import LPIPS
from nerfacc.estimators.occ_grid import OccGridEstimator
import numpy as np
import plotly.graph_objects as go
from skimage.metrics import structural_similarity as SSIM
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

# local imports
import core.models as M
from nerfdata.datasets import llff, blender
from nerfdata.utils.splitter import Splitter
import render.rendering as R
import utils.parser as P
from utils.camera3dplotter import Camera3DPlotter
from playground.model_trainers.nerf_trainer import NeRFModelTrainer
from playground.training_configuration import TrainingConfiguration

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
    dataset_name, dataset_kwargs = dataset_config[args.dataset]

    # get training, validation and test dataloaders
    splitter = Splitter(args.dataset, args.scene, n_training_views=args.n_imgs)
    splitter.split()
    datasets = splitter.get_datasets(train_img_mode=False, **dataset_kwargs)
    train_dataset, val_dataset, test_dataset = datasets

    if not args.debug:
        cam_plotter = create_camera_plotter(datasets)
        cam_plotter.upload_plot()

    if not args.render_only:
        model, lpips_net = init_models()

        training_settings = TrainingConfiguration(device, args)
        # TODO: Temporary workaround but probably need to move OccGridConfig one level up
        training_settings.occupancy_estimator_settings.aabb = train_dataset.aabb
        model_trainer = NeRFModelTrainer(training_settings, args.debug)

        # trains model using the trainer's configuration
        model_trainer.fit(model, train_dataset)

        # model_evaluator.evaluate(model, test_dataset)
        model.eval()
        estimator.eval()
        lpips_net.eval()
        lpips_net.to(device)
        with torch.no_grad():
            val_metrics = evaluation(
                train_dataset.hwf,
                model,
                estimator,
                lpips_net,
                test_dataset,
                2 * args.batch_size,
                device,
            )
        # log final metrics
        final_psnr, final_ssim, final_lpips = val_metrics

        if not args.debug:
            wandb.log(
                {
                    "final_psnr": final_psnr,
                    "final_ssim": final_ssim,
                    "final_lpips": final_lpips,
                }
            )
    else:
        model = init_models()
        # load model
        model.load_state_dict(torch.load(out_dir + "/model/nn.pt"))

    if not args.debug:
        # build base path for output directories
        out_dir = os.path.normpath(
            os.path.join(
                args.out_dir,
                args.model,
                args.dataset,
                args.scene,
                f"n_imgs_{str(args.n_imgs)}",
                run.id,
            )
        )

        # create output directories
        folders = ["video", "model"]
        [os.makedirs(os.path.join(out_dir, f), exist_ok=True) for f in folders]
        # save model
        if not args.render_only:
            torch.save(model.state_dict(), out_dir + "/model/nn.pt")

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
                "rgb_video": wandb.Video(frames, format="mp4", fps=30),
                "depth_video": wandb.Video(d_frames, format="mp4", fps=30),
            }
        )


def get_computing_device() -> torch.device:
    computing_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing device: {torch.cuda.get_device_name(computing_device)}")
    return computing_device


def init_wandb():
    wandb.login()
    # set up wandb run to track training
    name = f"{args.model}_{args.dataset}_img{args.n_imgs}_layer{args.n_layers}"
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


def init_models() -> Tuple[nn.Module, LPIPS]:
    """
    Initialize NeRF-like model and LPIPS net.
    ----------------------------------------------------------------------------
    Args:
        None
    Returns:
        Tuple[nn.Module, LPIPS]: models
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

    lpips_net = LPIPS(net="vgg")

    return model, lpips_net


def evaluation(
    hwf: Tuple[int, int, float],
    model: nn.Module,
    estimator: OccGridEstimator,
    lpips_net: LPIPS,
    dataset: Dataset,
    chunksize: int,
    device: torch.device,
    render_step_size: float = 5e-3,
) -> Tuple[float, float, float]:
    """
    Performs evaluation for NeRF-like model.
    ----------------------------------------------------------------------------
    """
    ndc = dataset.ndc
    rgbs = []
    rgbs_gt = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    for val_data in data_loader:
        rgb_gt, pose = val_data
        rgbs_gt.append(rgb_gt)  # append ground truth rgb
        rgb, _ = R.render_frame(
            hwf,
            data_loader.dataset.near,
            data_loader.dataset.far,
            pose[0],
            chunksize,
            estimator,
            model,
            train=False,
            ndc=ndc,
            white_bkgd=args.white_bkgd,
            render_step_size=render_step_size,
            device=device,
        )
        rgbs.append(rgb)  # append rendered rgb

    # compute PSNR
    rgbs = torch.permute(torch.stack(rgbs, dim=0), (0, 3, 1, 2))
    rgbs_gt = torch.permute(torch.cat(rgbs_gt, dim=0), (0, 3, 1, 2))
    rgbs_gt = rgbs_gt.to(device)
    val_psnr = -10.0 * torch.log10(F.mse_loss(rgbs, rgbs_gt))
    val_size = len(data_loader)

    # compute LPIPS
    if val_size < 25:
        val_lpips = lpips_net(rgbs, rgbs_gt).mean()
    else:
        # compute LPIPS in chunks
        n_chunks = 5
        chunk_size = val_size // n_chunks
        chunk_idxs = [i for i in range(0, val_size, chunk_size)]
        chunks = [
            (rgbs[i : i + chunk_size], rgbs_gt[i : i + chunk_size]) for i in chunk_idxs
        ]
        val_lpips = 0.0
        for chunk, chunk_gt in chunks:
            val_lpips += lpips_net(chunk, chunk_gt).mean()
        val_lpips /= n_chunks
    val_lpips = None

    # compute SSIM
    rgbs = torch.permute(rgbs, (0, 2, 3, 1)).cpu().numpy()
    rgbs_gt = torch.permute(rgbs_gt, (0, 2, 3, 1)).cpu().numpy()
    ssims = np.zeros((rgbs.shape[0],))
    val_ssim = 0.0
    for rgb, rgb_gt in zip(rgbs, rgbs_gt):
        val_ssim += SSIM(
            rgb, rgb_gt, channel_axis=-1, data_range=1.0, gaussian_weights=True
        )
    val_ssim /= len(rgbs)

    return val_psnr, val_ssim, val_lpips


if __name__ == "__main__":
    main()
