# standard library modules
import os

# third-party modules
import numpy as np
import yaml

class _FlowListDumper(yaml.SafeDumper):
    """YAML dumper that renders lists inline (flow style) while keeping
    mappings in block style."""


_FlowListDumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)

LLFF_BASE_PATH = os.path.normpath("../datasets/llff")
IMAGES_FOLDER = "images_8"
LLFF_HOLD = 8  # standard NeRF LLFF split: every 8th image is held out for eval
N_IMGS_CHOICES = (3, 6, 9)  # few-shot LLFF settings


def create_split_file(
    output_path: str = "../configs/split.yaml",
    dataset_path: str = LLFF_BASE_PATH,
    images_folder: str = IMAGES_FOLDER,
    n_imgs_choices: tuple = N_IMGS_CHOICES,
    llffhold: int = LLFF_HOLD,
) -> None:
    """
    Builds a YAML split file listing, for every LLFF scene found under
    `dataset_path`, the image indices to be used for evaluation and, for each
    few-shot setting in `n_imgs_choices`, the image indices to be used for
    training.

    Splitting rule (standard NeRF few-shot LLFF indexing):
        - Every `llffhold`-th image is held out for evaluation.
        - The remaining images form the training pool. For each n in
          `n_imgs_choices`, that pool is subsampled down to n evenly-spaced
          views (few-shot LLFF setting).

    Args:
        output_path (str): path where the YAML config file will be written.
        dataset_path (str): path to the LLFF dataset root, containing one
            folder per scene.
        images_folder (str): name of the per-scene folder containing images.
        n_imgs_choices (tuple): numbers of training views to build a split
            for.
        llffhold (int): holdout rate used to build the evaluation split.

    Output format:
        <scene>:
          eval: [idx, ...]
          train:
            n_imgs_<n_imgs>: [idx, ...]
    """
    assert os.path.isdir(
        dataset_path
    ), f"LLFF dataset folder {os.path.abspath(dataset_path)} not found."

    scenes = sorted(
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )

    split = {}
    for scene in scenes:
        imgs_folder_path = os.path.join(dataset_path, scene, images_folder)
        if not os.path.isdir(imgs_folder_path):
            continue

        n_available = sum(
            1
            for f in os.listdir(imgs_folder_path)
            if f.endswith(("JPG", "jpg", "png"))
        )
        if not n_available:
            continue

        all_idxs = np.arange(n_available)
        eval_idxs = all_idxs[all_idxs % llffhold == 0]
        train_pool = all_idxs[all_idxs % llffhold != 0]

        train_splits = {}
        for n_imgs in n_imgs_choices:
            train_idxs = train_pool
            if 0 < n_imgs < len(train_pool):
                sub_idxs = (
                    np.linspace(0, len(train_pool) - 1, n_imgs).round().astype(int)
                )
                train_idxs = train_pool[sub_idxs]

            train_splits[f"n_imgs_{n_imgs}"] = [int(i) for i in train_idxs]

        split[scene] = {
            "eval": [int(i) for i in eval_idxs],
            "train": train_splits,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(split, f, Dumper=_FlowListDumper, default_flow_style=False, sort_keys=False)


def load_train_split(
    scene: str,
    n_imgs: int,
    split_path: str = "../configs/split.yaml",
) -> list[int]:
    """
    Reads a split YAML file (as produced by `create_split_file`) and returns
    the training image indices for the given scene and number of training
    views.

    Args:
        scene (str): name of the LLFF scene to look up.
        n_imgs (int): few-shot training setting to look up (must match one of
            the `n_imgs_choices` the split file was built with).
        split_path (str): path to the split YAML file.

    Returns:
        list[int]: training image indices.
    """
    assert os.path.isfile(
        split_path
    ), f"Split file {os.path.abspath(split_path)} not found."

    with open(split_path, "r") as f:
        split = yaml.safe_load(f)

    assert scene in split, f"Scene '{scene}' not found in split file {split_path}."

    train_splits = split[scene]["train"]
    key = f"n_imgs_{n_imgs}"
    assert key in train_splits, (
        f"n_imgs={n_imgs} not found for scene '{scene}' in split file "
        f"{split_path}."
    )

    return train_splits[key]


def load_eval_split(
    scene: str,
    split_path: str = "../configs/split.yaml",
) -> list[int]:
    """
    Reads a split YAML file (as produced by `create_split_file`) and returns
    the evaluation image indices for the given scene.

    Args:
        scene (str): name of the LLFF scene to look up.
        split_path (str): path to the split YAML file.

    Returns:
        list[int]: evaluation image indices.
    """
    assert os.path.isfile(
        split_path
    ), f"Split file {os.path.abspath(split_path)} not found."

    with open(split_path, "r") as f:
        split = yaml.safe_load(f)

    assert scene in split, f"Scene '{scene}' not found in split file {split_path}."

    return split[scene]["eval"]
