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

Glue for the SC-CE pipeline: generate counterfactuals for a set of instances
and turn them into one distance score per instance.
"""
import time

import numpy as np

from .distances import aggregate, euclidean


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
