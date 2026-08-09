"""NeRF-like model definitions exposed through a single namespace."""

from core.models.nerf import Nerf
from core.models.sinerf import Sinerf

__all__ = ["Nerf", "Sinerf"]
