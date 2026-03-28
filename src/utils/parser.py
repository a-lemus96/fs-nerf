# stdlib imports
import argparse

def config_parser() -> argparse.Namespace:
    """Creates a parser for command-line arguments.
    ----------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(description="Train NeRF for view synthesis.")

    enc = parser.add_argument_group("Encoder")
    enc.add_argument("--d_input", default=3, type=int, help="Spatial input dimension")
    enc.add_argument(
        "--n_freqs",
        default=10,
        type=int,
        help="Number of encoding functions for spatial coordinates",
    )
    enc.add_argument(
        "--log_space",
        action="store_false",
        help="If not set, frequency scale is in logarithmic space",
    )
    enc.add_argument(
        "--no_dirs",
        action="store_true",
        help="If set, do not model view-dependent effects",
    )
    enc.add_argument(
        "--n_freqs_views",
        dest="n_freqs_views",
        default=4,
        type=int,
        help="Number of encoding fns for view dirs",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        choices=["nerf", "sinerf"],
        default="sinerf",
        help="Model to be used for training",
    )
    model.add_argument(
        "--d_filter", default=256, type=int, help="Linear layer filter dimension"
    )
    model.add_argument(
        "--n_layers", default=8, type=int, help="Number of layers preceding bottleneck"
    )
    model.add_argument(
        "--skip",
        default=[4],
        type=list,
        help="Layers at which to apply input residual connections",
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        choices=["synthetic", "llff"],
        default="llff",
        type=str,
        help="Dataset to choose scenes from",
    )
    data.add_argument(
        "--scene", default="flower", type=str, help="Scene to be used for training"
    )
    data.add_argument(
        "--n_imgs",
        default=100,
        type=int,
        help="Number of images to be used for training",
    )
    data.add_argument(
        "--img_mode",
        action="store_true",
        help="If set, iterate over images instead of rays for training",
    )
    data.add_argument(
        "--white_bkgd",
        action="store_true",
        help="Use white background for training imgs",
    )
    data.add_argument(
        "--factor", default=4, type=int, help="Downsample factor for LLFF dataset"
    )
    data.add_argument(
        "--bd_factor", default=0.75, type=float, help="Bound factor for LLFF dataset"
    )
    data.add_argument(
        "--no_recenter",
        action="store_true",
        help="If set, do not recenter LLFF dataset",
    )

    train = parser.add_argument_group("Training")
    train.add_argument(
        "--n_iters", default=20**3, type=int, help="Number of training iterations"
    )
    train.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="Number of rays per optimization step",
    )
    train.add_argument(
        "--lro", default=5e-4, type=float, help="Initial learning rate for optimizer"
    )
    train.add_argument(
        "--lrf", default=0.0, type=float, help="Final learning rate for optimizer"
    )
    train.add_argument(
        "--decay_rate",
        default=0.1,
        type=float,
        help="Decay rate for exponential learning rate scheduler",
    )
    train.add_argument(
        "--Td",
        default=250000,
        type=int,
        help="Number of iterations for learning rate decay",
    )
    train.add_argument(
        "--scheduler",
        choices=["const", "exp"],
        default="exp",
        help="Learning rate scheduler",
    )

    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument(
        "--val", action="store_true", help="If set, perform validation during training"
    )
    evaluation.add_argument(
        "--val_rate",
        default=500,
        type=int,
        help="Number of iterations between validation steps",
    )
    evaluation.add_argument(
        "--val_batch_size_multiplier",
        default=4,
        type=int,
        help="Multiplier applied to batch_size for evaluation chunking. "
        "Can be larger than training since no gradients are computed.",
    )
    evaluation.add_argument(
        "--val_ratio",
        default=0.25,
        type=float,
        help="Ratio of val data to be used in between epochs",
    )
    evaluation.add_argument(
        "--lpips_chunk_size",
        default=25,
        type=int,
        help="Number of images per chunk for LPIPS evaluation. Larger values use more GPU memory.",
    )

    regularizers = parser.add_argument_group("Regularizers")
    regularizers.add_argument(
        "--alpha",
        default=None,
        type=float,
        help="Initial alpha value for regularizing model parameter weights",
    )
    regularizers.add_argument(
        "--freq_scheduler",
        choices=["constant", "linear"],
        default="constant",
        help="Frequency regularizer scheduler type",
    )
    regularizers.add_argument(
        "--reg_ratio",
        default=0.5,
        type=float,
        help="Ratio of iterations for alpha regularizer scheduler",
    )
    regularizers.add_argument(
        "--reg",
        choices=["l1", "l2"],
        default="l1",
        help="Norm for penalizing model parameters",
    )
    regularizers.add_argument(
        "--beta",
        default=None,
        type=float,
        help="Occlusion regularization importance parameter",
    )
    regularizers.add_argument(
        "--depth_thres",
        default=0.2,
        type=float,
        help="Depth threshold to penalize sigma values",
    )

    logging = parser.add_argument_group("Logging")
    logging.add_argument(
        "--out_dir",
        default="../out/",
        type=str,
        help="Base directory for storing results",
    )

    debugging = parser.add_argument_group("Debugging")
    debugging.add_argument(
        "--debug", action="store_true", help="If set, run in debug mode"
    )

    rendering = parser.add_argument_group("Video Rendering")
    rendering.add_argument(
        "--render_only",
        action="store_true",
        help="If set, load pretrained model to render video",
    )

    args = parser.parse_args()

    return args
