# standard library modules
import os

# third-party modules
import imageio as iio
import numpy as np
from numpy import ndarray
from torch import Tensor


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector.
    ----------------------------------------------------------------------------
    Args:
        v (np.array): [N,]. Vector
    Returns:
        v (np.array): [N,]. Normalized vector
    """
    return v / np.linalg.norm(v)


def viewmatrix(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Computes the view matrix.
    ----------------------------------------------------------------------------
    Args:
        z (ndarray): [3,]. View direction
        up (ndarray): [3,]. Up direction
        pos (ndarray): [3,]. Camera position
    Returns:
        view (ndarray): [3, 4]. View matrix without bottom row
    """
    z = normalize(z)
    y = up
    x = normalize(np.cross(y, z))
    y = normalize(np.cross(z, x))
    matrix = np.stack([x, y, z, pos], axis=1)

    return matrix


def avg_pose(poses: np.ndarray) -> np.ndarray:
    """
    Computes camera to world matrix.
    ----------------------------------------------------------------------------
    Args:
        poses (Tensor): [N, 3, 5]. Camera poses
    Returns:
        c2w (Tensor): [N, 3, 5]. Camera to world matrix
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    viewdir = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(viewdir, up, center), hwf], 1)

    return c2w


def recenter_poses(poses: np.ndarray) -> np.ndarray:
    """
    Re-centers camera poses.
    ----------------------------------------------------------------------------
    Args:
        poses (Tensor): [N, 3, 5]. Camera poses
    Returns:
        poses (Tensor): [N, 3, 5]. Re-centered camera poses
    """
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])  # last row of camera matrix
    c2w = avg_pose(poses)  # average pose
    c2w = np.concatenate([c2w[:3, :4], bottom], axis=-2)  # center to world
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # camera to world

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_

    return poses


def postprocess_poses(
    poses: np.ndarray,
    bounds: np.ndarray,
    bd_factor: float = 0.75,
    recenter: bool = True,
) -> tuple[ndarray, tuple[int, int, float], float, float, ndarray]:
    """
    Rescales and re-centers the scene poses, and derives the bounds and
    intrinsics.
    ----------------------------------------------------------------------------
    Args:
        poses (ndarray): [N, 3, 5]. Camera poses with hwf in the last column
        bounds (ndarray): [N, 2]. Near and far bounds per view
        bd_factor (float): shrinks the scene so the nearest bound sits at
                           1 / bd_factor. None leaves the scale untouched
        recenter (bool): if True, expresses poses relative to the average pose
    Returns:
        poses (ndarray): [N, 3, 4]. Rescaled and re-centered camera poses
        hwf (Tuple[int, int, float]): image height, width and focal length
        min_bound (float): minimum value across the poses
        max_bound (float): maximum value across the poses
    """
    # rescale bounds and poses
    scale = 1.0 if bd_factor is None else 1.0 / (bounds.min() * bd_factor)
    poses[..., :3, 3] *= scale
    bounds *= scale

    if recenter:
        poses = recenter_poses(poses)

    hwf = poses[0, :3, -1]
    hwf = (int(hwf[0]), int(hwf[1]), float(hwf[2]))
    poses = poses[:, :3, :4]
    min_bound = poses.min()
    max_bound = poses.max()

    return poses, hwf, min_bound, max_bound


def load_scene(
    scene: str,
    bd_factor: float = 0.75,
    recenter: bool = True,
) -> tuple[ndarray, ndarray, tuple[int, int, float], float, float]:
    """
    Loads image paths, camera poses, bounds and intrinsics from an llff
    dataset folder.
    ----------------------------------------------------------------------------
    Expected scene folder structure:
        images_8/          -> contains image files
        poses_bounds.npy   -> poses file
    Args:
        scene (str): scene folder name under ../datasets/llff/
        bd_factor (float): shrinks the scene so the nearest bound sits at
                           1 / bd_factor. None leaves the scale untouched
        recenter (bool): if True, expresses poses relative to the average pose
    Returns:
        img_paths (ndarray): [N,]. Absolute paths to the scene images
        poses (ndarray): [N, 3, 4]. Camera poses
        hwf (Tuple[int, int, float]): image height, width and focal length
        min_bound (float): minimum value across the poses
        max_bound (float): maximum value across the poses
    """
    base_folder_path = os.path.normpath("../datasets/llff/")
    assert os.path.isdir(
        base_folder_path
    ), f"LLFF dataset folder {os.path.abspath(base_folder_path)} not found."
    all_scenes = os.listdir(path=base_folder_path)
    assert (
        scene in all_scenes
    ), f"Scene '{scene}' not found in local LLFF dataset folder."

    # load camera poses and bounds
    base_scene_folder_path = os.path.join(base_folder_path, scene)
    data = np.load(os.path.join(base_scene_folder_path, "poses_bounds.npy"))
    poses = data[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bounds = data[:, -2:].transpose([1, 0])

    # load the downsampled images
    imgs_folder_path = os.path.normpath(
        os.path.join(base_folder_path, scene, "images_8/")
    )
    assert os.path.isdir(
        imgs_folder_path
    ), f"Images folder path {os.path.abspath(imgs_folder_path)} not found."
    img_paths = [
        os.path.abspath(os.path.join(imgs_folder_path, f))
        for f in sorted(os.listdir(imgs_folder_path))
        if f.endswith(("JPG", "jpg", "png"))
    ]
    img_paths = np.array(img_paths)
    assert (
        len(img_paths) == poses.shape[-1]
    ), "Mismath between the number of images and poses"

    # modify camera poses
    H, W, _ = iio.imread(img_paths[0]).shape
    poses[:2, 4, :] = np.array([H, W]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / 8.0
    # correct poses ordering
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], axis=1
    )

    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)

    (poses, hwf, min_bound, max_bound) = postprocess_poses(
        poses, bounds, bd_factor, recenter
    )

    return img_paths, poses, hwf, min_bound, max_bound


def build_path(
    c2w: np.ndarray,
    poses: np.ndarray,
    bounds: np.ndarray,
    n_views: int = 60,
    n_rots: int = 2,
    zrate: float = 0.5,
    path_zflat: bool = False,
) -> Tensor:
    """
    Build spiral path for rendering sample video.
    ----------------------------------------------------------------------------
    """
    up = normalize(poses[:, :3, 1].sum(0))  # average up
    # compute reasonable focus depth for the scene
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # compute radii for spiral path
    shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)

    if path_zflat:
        zloc = -close_depth * 0.1
        c2w[:3, 3] = c2w[:3, 3] + zloc * c2w[:3, 2]
        rads[2] = 0.0
        n_rots = 1
        n_views /= 2

    # compute spiral path
    path_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_views + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(
            c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
        )
        path_poses.append(
            np.concatenate([viewmatrix(z, up, c), hwf], 1)
        )
    # cast to tensor
    return path_poses
