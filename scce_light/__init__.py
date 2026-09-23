"""
scce-light: lightweight SC-CE — selective classification via counterfactual explanations.
"""
from .growing_spheres import GrowingSpheres
from .distances import euclidean, aggregate
from .rejectors import PlugInRule, CFDistRejector
from .metrics import selective_metrics, evaluate
from .pipeline import generate_counterfactuals, cf_distances

__version__ = "0.1.0"

__all__ = [
    "GrowingSpheres",
    "euclidean", "aggregate",
    "PlugInRule", "CFDistRejector",
    "selective_metrics", "evaluate",
    "generate_counterfactuals", "cf_distances",
]
