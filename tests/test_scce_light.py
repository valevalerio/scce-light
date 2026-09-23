import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from scce_light import (GrowingSpheres, PlugInRule, CFDistRejector,
                        generate_counterfactuals, cf_distances, selective_metrics)


def _setup():
    X, y = make_classification(n_samples=300, n_features=4, random_state=0)
    model = LogisticRegression().fit(X, y)
    return X, y, model


def test_counterfactuals_flip_prediction():
    X, _, model = _setup()
    gs = GrowingSpheres(model.predict, random_state=0)
    for x in X[:10]:
        cfs = gs.generate(x, num_cf=5)
        assert cfs.shape == (5, X.shape[1])
        assert np.all(model.predict(cfs) != model.predict(x.reshape(1, -1)))


def test_distance_matches_linear_boundary():
    # for a linear model the closest counterfactual lies just beyond the hyperplane
    X, _, model = _setup()
    gs = GrowingSpheres(model.predict, random_state=0)
    d = cf_distances(X[:40], generate_counterfactuals(gs, X[:40], num_cf=4, verbose=False))
    true = np.abs(model.decision_function(X[:40])) / np.linalg.norm(model.coef_)
    assert np.all(d >= true - 1e-9)
    assert np.corrcoef(d, true)[0, 1] > 0.99


def test_rejectors_reach_target_coverage():
    X, y, model = _setup()
    coverages = [0.9, 0.7, 0.5]
    plg = PlugInRule(model).calibrate(X, coverages)
    scores = np.random.default_rng(0).gamma(2.0, size=len(X))
    cfd = CFDistRejector(model).calibrate(scores, coverages)
    for c in coverages:
        assert abs(plg.accept(X, c).mean() - c) < 0.02
        assert abs(cfd.accept(scores, c).mean() - c) < 0.02


def test_selective_metrics():
    m = selective_metrics(y_true=[0, 0, 1, 1], y_pred=[0, 1, 1, 0], accepted=[1, 0, 1, 1])
    assert m['coverage'] == 0.75
    assert np.isclose(m['nonrejected_accuracy'], 2 / 3)
    assert np.isclose(m['classification_quality'], 3 / 4)
