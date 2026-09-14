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
        "--decay_rate",
        default=0.1,
        type=float,
        help="Decay rate for exponential learning rate scheduler",
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
