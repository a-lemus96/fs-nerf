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
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# local imports
import core.models as M
import core.loss as L
import core.scheduler as S
from nerfdata.datasets import llff, blender
from nerfdata.utils.splitter import Splitter
import render.rendering as R
import utils.parser as P
import utils.plotting as PL

# GLOBAL VARIABLES
k = 0  # global step counter
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000  # memory snapshot

# RANDOM SEED
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args = P.config_parser()  # parse command line arguments

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# MEMORY SNAPSHOT
""" def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return
    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
    return """

# MODEL INITIALIZATION


def init_models(aabb: List[int]) -> Tuple[nn.Module, OccGridEstimator, LPIPS]:
    """
    Initialize NeRF-like model, occupancy grid estimator, and LPIPS net.
    ----------------------------------------------------------------------------
    Args:
        aabb (int): axis-aligned bounding box
    Returns:
        Tuple[nn.Module, OccGridEstimator, LPIPS]: models
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

    # model parameters
    grid_resolution = 128
    grid_nlvl = 1 if args.dataset == "synthetic" else 4
    # render parameters
    render_step_size = 5e-3
    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    )
    # initialize LPIPS network
    lpips_net = LPIPS(net="vgg")

    return model, estimator, lpips_net


# TRAINING FUNCTIONS


def validation(
    hwf: Tuple[int, int, float],
    model: nn.Module,
    estimator: OccGridEstimator,
    lpips_net: LPIPS,
    val_loader: DataLoader,
    chunksize: int,
    device: torch.device,
    render_step_size: float = 5e-3,
) -> Tuple[float, float, float]:
    """
    Performs validation step for NeRF-like model.
    ----------------------------------------------------------------------------
    Args:
        model (nn.Module): NeRF-like model
        estimator (OccGridEstimator): occupancy grid estimator
        lpips_net (LPIPS): LPIPS network
        val_loader (DataLoader): data loader
        chunksize (int): size of chunks for rendering frames
        device (torch.device): device to be used
        render_step_size (float, optional): step size for rendering
    Returns:
        val_psnr (float): validation PSNR
        val_ssim (float): validation SSIM
        val_lpips (float): validation LPIPS
    """
    H, W, focal = hwf
    ndc = val_loader.dataset.ndc
    rgbs = []
    rgbs_gt = []
    for val_data in val_loader:
        rgb_gt, pose = val_data
        rgbs_gt.append(rgb_gt)  # append ground truth rgb
        rgb, _ = R.render_frame(
            hwf,
            val_loader.dataset.near,
            val_loader.dataset.far,
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
    val_size = len(val_loader)

    # compute LPIPS
    """if val_size < 25:
        val_lpips = lpips_net(rgbs, rgbs_gt).mean()
    else:
        # compute LPIPS in chunks
        n_chunks = 5
        chunk_size = val_size //  n_chunks
        chunk_idxs = [i for i in range(0, val_size, chunk_size)]
        chunks = [(rgbs[i:i+chunk_size], rgbs_gt[i:i+chunk_size]) 
                  for i in chunk_idxs]
        val_lpips = 0.
        for chunk, chunk_gt in chunks:
            val_lpips += lpips_net(chunk, chunk_gt).mean()
        val_lpips /= n_chunks"""
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


def train(
    model: nn.Module,
    estimator: OccGridEstimator,
    lpips_net: LPIPS,
    train_loader: DataLoader,
    val_loader: DataLoader,
    render_step_size: float = 5e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Train NeRF model.
    ----------------------------------------------------------------------------
    Args:
        model (nn.Module): NeRF model
        estimator (OccGridEstimator): occupancy grid estimator
        lpips_net (LPIPS): LPIPS network
        train_loader (DataLoader): training set loader
        val_loader (DataLoader): validation set loader
        render_step_size (float, optional): step size for rendering
        device (torch.device): device to train on
    Returns:
        Tuple[float, float, float]: validation PSNR, SSIM, LPIPS
    ----------------------------------------------------------------------------
    """
    # retrieve camera intrinsics
    hwf = train_loader.dataset.hwf
    H, W, focal = hwf
    ndc = train_loader.dataset.ndc

    # set up optimizer and scheduler
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lro)
    sc_dict = {
        "const": (S.Constant, {}),
        "exp": (S.ExponentialDecay, {"r": args.decay_rate}),
    }
    class_name, kwargs = sc_dict[args.scheduler]
    scheduler = class_name(optimizer, args.n_iters, args.lro, **kwargs)
    """n_iters = args.n_iters
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[n_iters // 2, 
                        n_iters * 3 // 4, 
                        n_iters * 5 // 6, 
                        n_iters * 9 // 10], 
            gamma=0.33
    )"""
    pbar = tqdm(range(args.n_iters), desc=f"[NeRF]")  # set up progress bar
    iterator = iter(train_loader)  # data iterator

    # occlusion regularizer
    if args.beta is not None:
        occ_reg = L.OcclusionRegularizer(args.a, args.b, args.func)

    alpha = args.ao
    for k in pbar:  # loop over the number of iterations
        model.train()
        estimator.train()
        # get next batch of data
        try:
            rays_o, rays_d, rgb_gt = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            rays_o, rays_d, rgb_gt = next(iterator)

        # render rays
        (rgb, *_, extras), ray_indices, t_vals = R.render_rays(
            rays_o=rays_o,
            rays_d=rays_d,
            estimator=estimator,
            model=model,
            train=True,
            white_bkgd=args.white_bkgd,
            render_step_size=render_step_size,
            device=device,
        )

        # compute loss and PSNR
        rgb_gt = rgb_gt.to(device)
        loss = F.mse_loss(rgb, rgb_gt)
        with torch.no_grad():
            psnr = -10.0 * torch.log10(loss).item()

        # occlusion regularization
        if args.beta is not None:
            sigmas = extras["sigmas"]
            if len(sigmas) > 0:
                loss += occ_reg(sigmas, t_vals, ray_indices)

        # weight decay regularization
        if alpha is not None:
            freq_reg = torch.tensor(0.0).to(device)
            # linear decay schedule
            Ts = int(args.reg_ratio * args.Td)
            if k < Ts:
                for name, param in model.named_parameters():
                    if "weight" in name and param.shape[0] > 3:
                        if args.reg == "l1":
                            freq_reg += torch.abs(param).sum()
                        else:
                            freq_reg += torch.square(param).sum().sqrt()

                loss += alpha * freq_reg

        # backpropagate loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # define occupancy evaluation function
        def occ_eval_fn(x):
            return model(x) * render_step_size

        # update occupancy grid
        with torch.cuda.amp.autocast():
            estimator.update_every_n_steps(
                step=k, occ_eval_fn=occ_eval_fn, occ_thre=1e-2
            )

        # log metrics
        if not args.debug and k % args.val_rate != 0:
            wandb.log({"train_psnr": psnr, "lr": scheduler.lr, "alpha": alpha})

        # compute validation
        compute_val = k % args.val_rate == 0 and k > 0 and args.val
        if compute_val:
            model.eval()
            estimator.eval()
            # lpips_net.eval()
            with torch.no_grad():
                val_metrics = validation(
                    hwf,
                    model,
                    estimator,
                    lpips_net,
                    val_loader,
                    2 * args.batch_size,
                    device,
                )
                val_psnr, val_ssim, val_lpips = val_metrics
                # render test image
                rgb, depth = R.render_frame(
                    hwf,
                    train_loader.dataset.near,
                    train_loader.dataset.far,
                    testpose,
                    2 * args.batch_size,
                    estimator,
                    model,
                    train=False,
                    ndc=ndc,
                    white_bkgd=args.white_bkgd,
                    render_step_size=render_step_size,
                    device=device,
                )

                # log data to wandb
                if not args.debug:
                    wandb.log(
                        {
                            "train_psnr": psnr,
                            "lr": scheduler.lr,
                            "alpha": alpha,
                            "val_psnr": val_psnr,
                            "val_ssim": val_ssim,
                            "val_lpips": val_lpips,
                            "rgb": wandb.Image(rgb.cpu().numpy(), caption="RGB"),
                            "depth": wandb.Image(
                                PL.apply_colormap(depth.cpu().numpy()), caption="Depth"
                            ),
                        }
                    )
    return


def main():
    # select device
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    if not args.debug:
        wandb.login()
        # set up wandb run to track training
        name = f"{args.model}_{args.dataset}_img{args.n_imgs}_layer{args.n_layers}"
        run = wandb.init(project="fs-nerf", name=name, config=args)

    # set up dataset configuration
    dataset_config = {
        "synthetic": (blender.BlenderDataset, {"white_bkgd": args.white_bkgd}),
        "llff": (llff.LLFFDataset, {"white_bkgd": args.white_bkgd, "ndc": True}),
    }
    dataset_name, dataset_kwargs = dataset_config[args.dataset]

    # get training, validation and test dataloaders
    splitter = Splitter(args.dataset, args.scene, n_training_views=args.n_imgs)
    splitter.split()
    dataloaders = splitter.get_dataloaders(
        train_batch_size=args.batch_size, train_img_mode=False, **dataset_kwargs
    )
    train_loader, val_loader, test_loader = dataloaders

    # log interactive 3D plot of camera positions
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=splitter.poses[splitter.train_ids, 0, 3],
                y=splitter.poses[splitter.train_ids, 1, 3],
                z=splitter.poses[splitter.train_ids, 2, 3],
                mode="markers",
                marker=dict(size=7, opacity=0.8, color="black"),
                name="train",
            ),
            go.Scatter3d(
                x=splitter.poses[splitter.val_ids, 0, 3],
                y=splitter.poses[splitter.val_ids, 1, 3],
                z=splitter.poses[splitter.val_ids, 2, 3],
                mode="markers",
                marker=dict(size=7, opacity=0.8, color="red"),
                name="val",
            ),
            go.Scatter3d(
                x=splitter.poses[splitter.test_ids, 0, 3],
                y=splitter.poses[splitter.test_ids, 1, 3],
                z=splitter.poses[splitter.test_ids, 2, 3],
                mode="markers",
                marker=dict(size=7, opacity=0.8, color="blue"),
                name="test",
            ),
        ],
        layout=go.Layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                x=0.05,  # X-coordinate of the legend anchor
                y=0.95,  # Y-coordinate of the legend anchor
                xanchor="left",  # Anchor the left side of the legend to x
                yanchor="top",  # Anchor the top side of the legend to y
            ),
        ),
    )
    # set fixed axis scales
    t = 1 if args.dataset == "llff" else 5
    factor = 1 if args.dataset == "llff" else 0
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-t, t]),
            yaxis=dict(range=[-t, t]),
            zaxis=dict(range=[-t * factor, t]),
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        )
    )

    if not args.debug:
        wandb.log({"camera_positions": fig})

    if not args.render_only:
        # Start recording memory snapshot history
        # start_record_memory_history()
        # initialize modules
        model, estimator, lpips_net = init_models(train_loader.dataset.aabb)
        model.to(device)
        estimator.to(device)
        # lpips_net.to(device)
        # train model
        train(model, estimator, lpips_net, train_loader, val_loader, device=device)
        # Create the memory snapshot file
        # export_memory_snapshot()

        # Stop recording memory snapshot history
        # stop_record_memory_history()
        # final validation set and loader
        # compute final validation metrics
        model.eval()
        estimator.eval()
        lpips_net.eval()
        with torch.no_grad():
            val_metrics = validation(
                train_loader.dataset.hwf,
                model,
                estimator,
                lpips_net,
                val_loader,
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
    path_poses = train_loader.dataset.path_poses

    # render frames for poses
    model.eval()
    estimator.eval()
    output = R.render_path(
        path_poses,
        train_loader.train_set.hwf,
        train_loader.train_set.near,
        train_loader.train_set.far,
        2 * args.batch_size,
        model,
        estimator,
        ndc=train_loader.dataset.ndc,
        white_bkgd=args.white_bkgd,
        device=device,
    )
    frames, d_frames = output

    if not args.debug:
        # put together frames and save result into .mp4 file
        R.render_video(basedir=f"{out_dir}/video/", frames=frames, d_frames=d_frames)
        # log final video renderings to wandb
        wandb.log(
            {
                "rgb_video": wandb.Video(f"{out_dir}/video/rgb.mp4", fps=30),
                "depth_video": wandb.Video(f"{out_dir}/video/depth.mp4", fps=30),
            }
        )


if __name__ == "__main__":
    main()
