# standard library modules
from collections.abc import Iterable
import logging
import os

# third-party modules
import numpy as np
import yaml

# custom imports
from utils.scene import load_scene

logger = logging.getLogger(__name__)

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
POSES_FILE = "poses_bounds.npy"
LLFF_HOLD = 8  # standard NeRF LLFF split: every 8th image is held out for eval
N_IMGS_CHOICES = (3, 6, 9)  # few-shot LLFF settings


def _validate_scene(
    scene: str,
    imgs_folder_path: str,
    poses_file_path: str,
) -> np.ndarray | None:
    """
    Validates that a scene has usable images and poses, logging a warning and
    returning None for the first failing check found.

    Returns:
        np.ndarray | None: poses, or None if the scene should be skipped.
    """
    skip_msg = "Skipping scene '%s': %s."

    if not os.path.isdir(imgs_folder_path):
        logger.warning(
            skip_msg, scene, f"missing images folder '{imgs_folder_path}'"
        )
        return None

    if not os.path.isfile(poses_file_path):
        logger.warning(
            skip_msg, scene, f"missing poses file '{poses_file_path}'"
        )
        return None

    img_paths, poses, *_ = load_scene(scene)
    n_imgs = len(img_paths)
    if not n_imgs:
        logger.warning(
            skip_msg, scene, f"no images found in '{imgs_folder_path}'"
        )
        return None

    if len(poses) == 0:
        logger.warning(
            skip_msg, scene, f"no poses found in '{poses_file_path}'"
        )
        return None

    if n_imgs != len(poses):
        logger.warning(
            skip_msg,
            scene,
            f"number of images ({n_imgs}) does not match number of "
            f"poses ({len(poses)})",
        )
        return None

    return poses

def _get_monitoring_id(
    poses: np.ndarray,
    train_idxs: Iterable,
    eval_idxs: Iterable,
) -> int | None:
    """
    Picks, among the leftover images (neither train nor eval), the one whose
    difficulty is most typical of the evaluation set.

    Difficulty is proxied by distance to the nearest training camera centre:
    letting d(i) be that distance for image i, the monitoring index m is
        m = argmin over leftover idxs of |d(m) - median{d(e) : e in eval_idxs}|,
    with ties broken by lowest index.
    ----------------------------------------------------------------------------
    Args:
        poses (np.ndarray): [N, 3, 4]. Camera poses for the full scene.
        train_idxs (Iterable): indices of images used for training.
        eval_idxs (Iterable): indices of images held out for evaluation.
    Returns:
        int | None: index of the image to use for monitoring, or None if no
            leftover image is available (train and eval idxs cover all poses).
    """
    all_idxs = set(range(len(poses)))
    train_idxs, eval_idxs = set(train_idxs), set(eval_idxs)
    assert train_idxs.isdisjoint(eval_idxs), "Train and eval idxs must be disjoint."
    assert (train_idxs | eval_idxs) <= all_idxs, (
        "Train and eval idxs must be in range [0, N_poses)."
    )

    idx_pool = sorted(all_idxs - (train_idxs | eval_idxs))
    if not idx_pool:
        logger.warning(
            "No leftover images available to pick a monitoring index from "
            "(train and eval idxs cover all %d poses).", len(poses),
        )
        return None

    monitor_origins = poses[idx_pool, :, -1]
    train_origins = poses[sorted(train_idxs), :, -1]
    eval_origins = poses[sorted(eval_idxs), :, -1]

    diff_m = monitor_origins[:, None, :] - train_origins[None, :, :] # [M, T, 3] diff array
    dist_m = np.linalg.norm(diff_m, axis=-1) # [M, T] dist array
    nearest_dist_m = np.min(dist_m, axis=-1) # [M,] dist array

    diff_e = eval_origins[:, None, :] - train_origins[None, :, :]
    dist_e = np.linalg.norm(diff_e, axis=-1)
    nearest_dist_e = np.min(dist_e, axis=-1)

    median = np.median(nearest_dist_e)

    monitor_idx = idx_pool[np.argmin(np.abs(nearest_dist_m - median))]

    return monitor_idx

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
        - For each such split, one leftover image (neither train nor eval)
          is picked as the monitoring image: the one whose distance to the
          nearest training view is closest to the eval set's median such
          distance (see `_get_monitoring_id`). This is null when no leftover
          image remains (train and eval idxs cover all poses).

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
          eval: [idxA, ...]
          train:
            n_imgs_<n_imgs>: [idxB, ...]
          monitor:
            n_imgs_<n_imgs>: idxC | null
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
        scene_path = os.path.join(dataset_path, scene)
        imgs_folder_path = os.path.join(scene_path, images_folder)
        poses_file_path = os.path.join(scene_path, POSES_FILE)

        poses = _validate_scene(scene, imgs_folder_path, poses_file_path)
        if poses is None:
            continue
        n_available = len(poses)

        all_idxs = np.arange(n_available)
        eval_idxs = [int(i) for i in all_idxs[all_idxs % llffhold == 0]]
        train_pool = all_idxs[all_idxs % llffhold != 0]

        train_splits = {}
        monitor_ids = {}
        for n_imgs in n_imgs_choices:
            train_idxs = train_pool
            if 0 < n_imgs < len(train_pool):
                sub_idxs = (
                    np.linspace(0, len(train_pool) - 1, n_imgs).round().astype(int)
                )
                train_idxs = train_pool[sub_idxs]

            key = f"n_imgs_{n_imgs}"
            train_splits[key] = [int(i) for i in train_idxs]
            monitor_ids[key] = _get_monitoring_id(poses, train_idxs, eval_idxs)

        split[scene] = {
            "eval": eval_idxs,
            "train": train_splits,
            "monitor": monitor_ids,
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
