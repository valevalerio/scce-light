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

Distances between an instance and its counterfactuals, and their per-instance aggregation.
"""
import numpy as np

AGGREGATIONS = ('min', 'mean', 'max')


def euclidean(cfs, x):
    """L2 distance between each row of ``cfs`` (k, d) and the instance ``x`` (d,)."""
    return np.linalg.norm(np.asarray(cfs) - np.asarray(x).reshape(1, -1), axis=1)


def aggregate(distances, how='min'):
    """Summarise the distances of one instance's counterfactuals; +inf if none were found."""
    distances = np.asarray(distances)
    if distances.size == 0:
        return np.inf
    return {'min': np.min, 'mean': np.mean, 'max': np.max}[how](distances)
