"""
Selective classifiers: a probabilistic model plus a reject option.

Both rejectors share the same mechanism: a score is computed for each instance
(higher = more trustworthy), thresholds are calibrated as quantiles of the
calibration scores, and an instance is accepted at coverage ``c`` iff its score
is at least the ``(1 - c)``-quantile.

- ``PlugInRule``: score = model confidence, max_k p(k|x) (Herbei & Wegkamp baseline).
- ``CFDistRejector``: score = distance to the counterfactuals (SC-CE).
  Instances whose counterfactuals are close lie near the decision boundary and are rejected.
"""
import numpy as np
from scipy import stats


class _QuantileRejector:
    def __init__(self, model):
        self.model = model
        self.thresholds_ = None

    def _calibrate_scores(self, scores, coverages, use_gamma=False):
        scores = np.asarray(scores, dtype=float)
        self.coverages_ = np.asarray(coverages, dtype=float)
        quantiles = 1 - self.coverages_
        if use_gamma:
            finite = scores[np.isfinite(scores)]
            if np.std(finite) < 1e-6:
                # degenerate case: all scores (almost) equal, a gamma cannot be fitted
                self.thresholds_ = np.quantile(scores, quantiles)
            else:
                a, loc, scale = stats.gamma.fit(finite)
                self.thresholds_ = stats.gamma.ppf(quantiles, a, loc=loc, scale=scale)
        else:
            self.thresholds_ = np.quantile(scores, quantiles)
        return self

    def _accept_scores(self, scores, coverage):
        if self.thresholds_ is None:
            raise ValueError("The rejector is not calibrated yet. Call calibrate() first.")
        idx = np.flatnonzero(np.isclose(self.coverages_, coverage))
        if idx.size == 0:
            raise ValueError(f"Coverage {coverage} was not among the calibrated coverages.")
        return np.asarray(scores) >= self.thresholds_[idx[0]]

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class PlugInRule(_QuantileRejector):
    """Confidence-based rejector (baseline)."""

    def score(self, X):
        return self.model.predict_proba(X).max(axis=1)

    def calibrate(self, X_cal, coverages):
        return self._calibrate_scores(self.score(X_cal), coverages)

    def accept(self, X, coverage):
        """Boolean mask: True = the model predicts, False = reject."""
        return self._accept_scores(self.score(X), coverage)


class CFDistRejector(_QuantileRejector):
    """
    Counterfactual-distance rejector (SC-CE).

    The distances are computed beforehand (see ``scce_light.pipeline.cf_distances``)
    because counterfactual generation is the expensive step.

    Parameters
    ----------
    model : fitted probabilistic classifier
    use_gamma : bool
        If True, thresholds are quantiles of a Gamma distribution fitted on the
        calibration distances instead of the empirical quantiles.
    """

    def __init__(self, model, use_gamma=False):
        super().__init__(model)
        self.use_gamma = use_gamma

    def calibrate(self, distances_cal, coverages):
        return self._calibrate_scores(distances_cal, coverages, use_gamma=self.use_gamma)

    def accept(self, distances, coverage):
        """Boolean mask: True = the model predicts, False = reject."""
        return self._accept_scores(distances, coverage)
