"""AH-GNO package interface."""

from .model import LearnedHorizonGNO, build_model
from .preprocessing import AreaWeightedNormalizer, compute_node_areas, normalize_coordinates
from .losses import AreaWeightedHuberLoss, compute_h_star, denormalize_dz

__all__ = [
    "LearnedHorizonGNO",
    "build_model",
    "AreaWeightedNormalizer",
    "compute_node_areas",
    "normalize_coordinates",
    "AreaWeightedHuberLoss",
    "compute_h_star",
    "denormalize_dz",
]

__version__ = "0.1.0"
