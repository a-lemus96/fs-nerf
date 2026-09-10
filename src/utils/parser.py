# stdlib imports
import argparse

def config_parser() -> argparse.Namespace:
    """Creates a parser for command-line arguments.
    ----------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(description="Train NeRF for view synthesis.")

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        choices=["nerf", "sinerf"],
        default="sinerf",
        help="Model to be used for training",
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--scene", default="flower", type=str, help="LLFF scene to be used for training"
    )
    data.add_argument(
        "--n_imgs",
        choices=[3, 6, 9],
        default=3,
        type=int,
        help="Number of training views (FreeNeRF few-shot LLFF settings)",
    )
    data.add_argument(
        "--img_mode",
        action="store_true",
        help="If set, iterate over images instead of rays for training",
    )
    data.add_argument(
        "--factor", default=4, type=int, help="LLFF dataset downsample factor"
    )
    data.add_argument(
        "--bd_factor", default=0.75, type=float, help="LLFF dataset bound factor"
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
