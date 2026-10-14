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

    model.add_argument(
        "--omega",
        default=30.0,
        type=float,
        help="SIREN omega (w_0) of the first layer; only used by sinerf",
    )

    freq = parser.add_argument_group("Frequency curriculum")
    freq.add_argument(
        "--freq_curriculum",
        choices=["none", "omega", "cap"],
        default="none",
        help="Frequency curriculum over the first SIREN layer: anneal omega, "
        "or cap each neuron's frequency omega * ||w_i||; only used by sinerf",
    )
    freq.add_argument(
        "--freq_start",
        default=None,
        type=float,
        help="Initial omega (omega) or frequency cutoff in rad/unit (cap); "
        "required when a curriculum is set",
    )
    freq.add_argument(
        "--freq_end",
        default=None,
        type=float,
        help="Final omega (omega) or cutoff (cap); defaults to --omega (omega) "
        "or the initial max neuron frequency (cap)",
    )
    freq.add_argument(
        "--freq_ramp",
        default=0.9,
        type=float,
        help="Fraction of training iterations over which the curriculum ramps; "
        "the cap is released afterwards",
    )
    freq.add_argument(
        "--freq_schedule",
        choices=["linear", "log"],
        default="log",
        help="Interpolation from start to end: linear or geometric",
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--scene", default="fern", type=str, help="LLFF scene to be used for training"
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
        "--n_iters",
        default=None,
        type=int,
        help="Number of training iterations; falls back to training.yaml if not set",
    )
    train.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Initial learning rate for optimizer; overrides the per-model default in training.yaml if set",
    )
    train.add_argument(
        "--lr_scheduler_type",
        default=None,
        type=str,
        help="Learning rate scheduler type (exp or constant); falls back to training.yaml if not set",
    )
    train.add_argument(
        "--decay_rate",
        default=None,
        type=float,
        help="Exponential decay rate for the lr scheduler; falls back to training.yaml if not set",
    )
    train.add_argument(
        "--warmup_iters",
        default=None,
        type=int,
        help="Lr warmup length in iterations for the exp scheduler; 0 disables "
        "warmup; falls back to training.yaml if not set",
    )
    train.add_argument(
        "--warmup_mult",
        default=None,
        type=float,
        help="Lr multiplier at the start of warmup; falls back to training.yaml if not set",
    )

    train.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for torch, numpy, random and the dedicated generators",
    )

    train.add_argument(
        "--tf32",
        action="store_true",
        help="If set, allow TF32 tensor-core matmuls (Ampere/Ada GPUs only)",
    )

    evaluation = parser.add_argument_group("Evaluation")
    evaluation.add_argument(
        "--val_every",
        default=None,
        type=int,
        help="Number of iterations between validation steps; falls back to evaluation.yaml if not set",
    )

    logging = parser.add_argument_group("Logging")
    logging.add_argument(
        "--out_dir",
        default="../output/",
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
