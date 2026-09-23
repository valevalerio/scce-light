"""
Glue for the SC-CE pipeline: generate counterfactuals for a set of instances
and turn them into one distance score per instance.
"""
import time
import numpy as np

from .distances import euclidean, aggregate


def generate_counterfactuals(generator, X, num_cf=8, verbose=True):
    """Run ``generator.generate`` on every row of X; returns a list of (k, d) arrays."""
    X = np.asarray(X, dtype=float)
    start = time.time()
    cfs = [generator.generate(x, num_cf=num_cf) for x in X]
    if verbose:
        print(f"Generated {num_cf} counterfactuals for {len(X)} instances "
              f"in {time.time() - start:.1f}s")
    return cfs


def cf_distances(X, cfs, distance=euclidean, how='min'):
    """One aggregated distance per instance: ``how`` in {'min', 'mean', 'max'}."""
    X = np.asarray(X, dtype=float)
    return np.array([aggregate(distance(c, x), how) for x, c in zip(X, cfs)])
