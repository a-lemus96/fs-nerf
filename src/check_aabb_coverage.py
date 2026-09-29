# stdlib imports
import argparse
import csv
import logging
import os

# third-party imports
import torch
from torch import Tensor
import yaml

# local imports
from llff import LLFFDataset
from utils import get_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLFF_BASE_PATH = os.path.normpath("../datasets/llff")
RENDERING_CONFIG_PATH = "../configs/rendering.yaml"


def discover_scenes(dataset_path: str = LLFF_BASE_PATH) -> list[str]:
    """Lists LLFF scene folder names under the dataset root, sorted."""
    assert os.path.isdir(
        dataset_path
    ), f"LLFF dataset folder {os.path.abspath(dataset_path)} not found."
    return sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )


def default_num_samples(near: float, far: float) -> int:
    """
    Derives a default per-ray sample count M from the occupancy grid's own
    render_step_size (configs/rendering.yaml), so the coverage check probes
    rays at roughly the same granularity nerfacc uses when marching them
    during training/rendering.
    """
    with open(RENDERING_CONFIG_PATH, "r") as f:
        render_step_size = yaml.safe_load(f)["render_step_size"]

    return max(2, round((far - near) / render_step_size))


def sample_points(rays_o: Tensor, rays_d: Tensor, near: float, far: float, M: int) -> Tensor:
    """
    Computes M evenly-spaced sample points per ray along t in [near, far],
    the same x = o + t * d parametrization used by Renderer.render_rays's
    sigma_fn/rgb_sigma_fn.
    ----------------------------------------------------------------------------
    Args:
        rays_o (Tensor): (R, 3). Ray origins (NDC space).
        rays_d (Tensor): (R, 3). Ray directions (NDC space).
        near (float):    t value of the near bound.
        far (float):     t value of the far bound.
        M (int):         number of samples per ray.
    Returns:
        pts (Tensor): (R, M, 3). Sample points along each ray.
    """
    t = torch.linspace(near, far, M, device=rays_o.device)  # (M,)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]  # (R, M, 3)

    return pts


def count_outside_aabb(pts: Tensor, aabb: Tensor) -> tuple[int, int]:
    """
    Counts how many sample points fall outside an axis-aligned bounding box.
    ----------------------------------------------------------------------------
    Args:
        pts (Tensor):  (..., 3). Sample points.
        aabb (Tensor): (6,). [xmin, ymin, zmin, xmax, ymax, zmax].
    Returns:
        n_outside (int): number of points outside the box.
        n_total (int):   total number of points.
    """
    lo, hi = aabb[:3], aabb[3:]
    outside = ((pts < lo) | (pts > hi)).any(dim=-1)

    return int(outside.sum().item()), int(outside.numel())


def scene_coverage(
    scene: str,
    aabb: Tensor,
    M: int,
    device: torch.device,
    ray_chunk_size: int,
) -> dict:
    """Loads a scene's full ray set and reports the fraction of per-ray
    samples that fall outside `aabb`."""
    dataset = LLFFDataset(scene)  # img_ids=None -> every image in the scene

    rays_o = dataset.rays_o.reshape(-1, 3)  # (N * H * W, 3)
    rays_d = dataset.rays_d.reshape(-1, 3)

    n_outside, n_total = 0, 0
    for chunk_o, chunk_d in zip(
        get_chunks(rays_o, ray_chunk_size), get_chunks(rays_d, ray_chunk_size)
    ):
        pts = sample_points(
            chunk_o.to(device), chunk_d.to(device), dataset.near, dataset.far, M
        )
        chunk_outside, chunk_total = count_outside_aabb(pts, aabb)
        n_outside += chunk_outside
        n_total += chunk_total

    return {
        "scene": scene,
        "n_images": len(dataset),
        "n_rays": rays_o.shape[0],
        "M": M,
        "total_samples": n_total,
        "samples_outside_aabb": n_outside,
        "pct_outside_aabb": 100.0 * n_outside / n_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reports, per LLFF scene, the percentage of per-ray "
        "samples that fall outside a given aabb."
    )
    parser.add_argument(
        "--aabb",
        nargs=6,
        type=float,
        default=[-1.2, -1.2, -1.0, 1.2, 1.2, 1.0],
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Axis-aligned bounding box to test against.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="Samples per ray. Defaults to (far - near) / render_step_size "
        "from rendering.yaml.",
    )
    parser.add_argument(
        "--ray_chunk_size",
        type=int,
        default=50_000,
        help="Rays processed at a time, to bound peak memory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../out/aabb_coverage.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)

    scenes = discover_scenes()
    if len(scenes) != 8:
        logger.warning("Expected 8 LLFF scenes, found %d: %s", len(scenes), scenes)

    # near/far are the same (0.0, 1.0) NDC bounds for every LLFF scene, but
    # M's default is only knowable once rendering.yaml has been read, so
    # resolve it against the first scene's dataset bounds.
    M = args.M

    rows = []
    with torch.no_grad():
        for scene in scenes:
            logger.info("Processing scene '%s'...", scene)
            if M is None:
                probe = LLFFDataset(scene, img_ids=[0])
                M = default_num_samples(probe.near, probe.far)
                logger.info("Derived default M=%d from render_step_size", M)

            result = scene_coverage(scene, aabb, M, device, args.ray_chunk_size)
            rows.append(result)
            logger.info(
                "%s: %.4f%% of %d samples outside aabb",
                scene, result["pct_outside_aabb"], result["total_samples"],
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote results to %s", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
