

"""
Copyright 2026 Valerio Bonsignori

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

scce-light: lightweight SC-CE — selective classification via counterfactual explanations.
"""
from .distances import aggregate, euclidean
from .growing_spheres import GrowingSpheres
from .metrics import evaluate, selective_metrics
from .pipeline import cf_distances, generate_counterfactuals
from .rejectors import CFDistRejector, PlugInRule

__version__ = "0.1.0"

__all__ = [
    "CFDistRejector",
    "GrowingSpheres",
    "PlugInRule",
    "aggregate",
    "cf_distances",
    "euclidean",
    "evaluate",
    "generate_counterfactuals",
    "selective_metrics",
]
