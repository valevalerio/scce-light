"""
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
