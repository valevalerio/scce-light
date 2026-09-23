"""
Metrics for classification with a reject option (Condessa et al., 2017).
"""
import numpy as np
import pandas as pd


def rejection_report(y_true, y_pred, accepted):
    """Counts of correct / misclassified instances among accepted and rejected ones."""
    correct = np.asarray(y_true) == np.asarray(y_pred)
    accepted = np.asarray(accepted, dtype=bool)
    return {
        'correct_nonrejected': int(np.sum(correct & accepted)),
        'correct_rejected': int(np.sum(correct & ~accepted)),
        'miscl_nonrejected': int(np.sum(~correct & accepted)),
        'miscl_rejected': int(np.sum(~correct & ~accepted)),
    }


def nonrejected_accuracy(correct_nonrejected, miscl_nonrejected, **_):
    """Accuracy on the accepted instances (0 if everything is rejected)."""
    n_acc = correct_nonrejected + miscl_nonrejected
    return correct_nonrejected / n_acc if n_acc > 0 else 0.0


def classification_quality(correct_nonrejected, correct_rejected, miscl_nonrejected, miscl_rejected):
    """Fraction of correct decisions: correct accepted + misclassified rejected."""
    n = correct_nonrejected + correct_rejected + miscl_nonrejected + miscl_rejected
    return (correct_nonrejected + miscl_rejected) / n


def rejection_quality(correct_nonrejected, correct_rejected, miscl_nonrejected, miscl_rejected):
    """
    Ratio between the error/correct odds among rejected instances and overall.
    > 1 means the rejector concentrates errors in the rejected set (1 = random rejection).
    """
    if correct_rejected == 0:
        return 0.0
    frac_rej = miscl_rejected / correct_rejected
    frac_all = (miscl_rejected + miscl_nonrejected) / (correct_rejected + correct_nonrejected)
    return frac_rej / frac_all if frac_all != 0 else 0.0


def selective_metrics(y_true, y_pred, accepted):
    """All selective metrics for one accept/reject mask."""
    r = rejection_report(y_true, y_pred, accepted)
    return {
        'coverage': float(np.mean(accepted)),
        'nonrejected_accuracy': nonrejected_accuracy(**r),
        'classification_quality': classification_quality(**r),
        'rejection_quality': rejection_quality(**r),
    }


def evaluate(rejector, y_true, y_pred, scores_or_X, coverages, name=None):
    """
    Evaluate a calibrated rejector over a list of target coverages.

    ``scores_or_X`` is what ``rejector.accept`` expects: X for ``PlugInRule``,
    the counterfactual distances for ``CFDistRejector``.
    """
    rows = []
    for c in coverages:
        accepted = rejector.accept(scores_or_X, c)
        rows.append({'rejector': name or type(rejector).__name__, 'target_coverage': c,
                     **selective_metrics(y_true, y_pred, accepted)})
    return pd.DataFrame(rows)
