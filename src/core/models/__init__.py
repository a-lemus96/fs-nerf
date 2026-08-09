"""NeRF-like model definitions exposed through a single namespace."""

from core.models.nerf import NeRF
from core.models.sinerf import SiNeRF

__all__ = ["NeRF", "SiNeRF"]
