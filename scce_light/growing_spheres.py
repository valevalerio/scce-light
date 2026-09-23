"""
Growing Spheres counterfactual generator (Laugel et al., 2018).

Self-contained re-implementation of the fork used by SC-CE
(https://github.com/valevalerio/growingspheres), including its
``num_enemies`` extension that returns several counterfactuals per instance.

Algorithm, for an instance x predicted as class c:
1. Zoom in: sample a ball of radius r around x; while it contains enemies
   (points not predicted as c) halve the radius (divide by ``decrease_radius``).
2. Grow: sample layers of increasing radius until one contains enemies.
3. Keep the ``num_enemies`` enemies closest to x (Euclidean).
4. Sparsify (optional): for each enemy, greedily reset to x the coordinates
   with the smallest change, as long as the prediction stays different from c.
"""
import numpy as np
from sklearn.utils import check_random_state


def _sample_ball(center, radius, n, rng):
    """Uniform samples inside a d-ball (via the (d+2)-sphere projection trick)."""
    d = center.shape[1]
    u = rng.normal(0, 1, (n, d + 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u[:, :d] * radius + center


def _sample_sphere(center, radius, n, rng):
    """Uniform samples on the surface of a d-sphere."""
    z = rng.normal(0, 1, (n, center.shape[1]))
    return z / np.linalg.norm(z, axis=1, keepdims=True) * radius + center


def _sample_ring(center, segment, n, rng):
    """Uniform samples in the shell segment[0] <= ||z - center|| <= segment[1]."""
    d = center.shape[1]
    z = rng.normal(0, 1, (n, d))
    r = rng.uniform(segment[0] ** d, segment[1] ** d, n) ** (1.0 / d)
    return z / np.linalg.norm(z, axis=1, keepdims=True) * r[:, None] + center


class GrowingSpheres:
    """
    Growing Spheres counterfactual generator.

    Parameters
    ----------
    predict_fn : callable
        Function mapping an (n, d) array to integer class labels (e.g. ``model.predict``).
    n_in_layer : int
        Number of points sampled in each layer.
    first_radius : float
        Radius of the first ball explored around the instance.
    decrease_radius : float
        Shrinking factor (> 1) of the zoom-in phase; also sets the growth step.
    layer_shape : {'ball', 'ring', 'sphere'}
        Shape of the layers explored in the growing phase.
    sparse : bool
        Apply the feature-selection (sparsification) step.
    max_iter : int
        Safety cap on the number of growing iterations.
    random_state : int, RandomState or None

    Example
    -------
    >>> gs = GrowingSpheres(model.predict, random_state=0)
    >>> cfs = gs.generate(x, num_cf=8)   # (8, d) array of counterfactuals
    """

    def __init__(self, predict_fn, n_in_layer=200, first_radius=0.05, decrease_radius=2.0,
                 layer_shape='ball', sparse=True, max_iter=10_000, random_state=None):
        if decrease_radius <= 1.0:
            raise ValueError("decrease_radius must be > 1.0")
        if layer_shape not in ('ball', 'ring', 'sphere'):
            raise ValueError("layer_shape must be 'ball', 'ring' or 'sphere'")
        self.predict_fn = predict_fn
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.decrease_radius = decrease_radius
        self.layer_shape = layer_shape
        self.sparse = sparse
        self.max_iter = max_iter
        self.rng = check_random_state(random_state)

    def generate(self, x, num_cf=8):
        """
        Return an array of shape (num_cf, d) with counterfactuals of ``x``.

        As in the SC-CE fork, the search is repeated until at least ``num_cf``
        enemies have been collected.
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        y_obs = self.predict_fn(x)[0]
        found = []
        while len(found) < num_cf:
            enemies = self._explore(x, y_obs)
            order = np.argsort(np.linalg.norm(enemies - x, axis=1))
            closest = enemies[order[:num_cf - len(found)]]
            if self.sparse:
                closest = self._feature_selection(x, y_obs, closest)
            found.extend(closest)
        return np.asarray(found)

    def _enemies_in(self, layer, y_obs):
        return layer[self.predict_fn(layer) != y_obs]

    def _explore(self, x, y_obs):
        """Find a layer containing enemies, starting from an enemy-free ball."""
        radius = self.first_radius
        # zoom in until the ball around x contains no enemies
        while len(self._enemies_in(_sample_ball(x, radius, self.n_in_layer, self.rng), y_obs)) > 0:
            radius /= self.decrease_radius
        radius /= self.decrease_radius
        step = radius / self.decrease_radius

        # grow until a layer contains enemies
        for _ in range(self.max_iter):
            if self.layer_shape == 'ring':
                layer = _sample_ring(x, (radius, radius + step), self.n_in_layer, self.rng)
            elif self.layer_shape == 'sphere':
                layer = _sample_sphere(x, radius + step, self.n_in_layer, self.rng)
            else:
                layer = _sample_ball(x, radius + step, self.n_in_layer, self.rng)
            enemies = self._enemies_in(layer, y_obs)
            if len(enemies) > 0:
                return enemies
            radius += step
        raise RuntimeError(f"No counterfactual found within {self.max_iter} iterations "
                           f"(final radius {radius:.3g}).")

    def _feature_selection(self, x, y_obs, enemies):
        """
        Greedy sparsification of each enemy: visit its coordinates by increasing
        |enemy - x| and reset each one to x while the prediction stays an enemy.
        Vectorised across enemies (one model call per coordinate rank).
        """
        out = enemies.copy()
        moves = np.abs(enemies - x)
        order = np.argsort(moves, axis=1, kind='stable')
        rows = np.arange(len(out))
        for rank in range(x.shape[1]):
            cols = order[:, rank]
            candidates = out.copy()
            candidates[rows, cols] = x[0, cols]
            keep = (self.predict_fn(candidates) != y_obs) & (moves[rows, cols] > 0)
            out[keep] = candidates[keep]
        return out
