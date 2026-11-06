import numpy as np


def _rate(values, mask):
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def statistical_parity_difference(y_pred, group):
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    return _rate(y_pred, group == 1) - _rate(y_pred, group == 0)


def disparate_impact_ratio(y_pred, group):
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    protected = _rate(y_pred, group == 1)
    reference = _rate(y_pred, group == 0)
    if reference == 0.0:
        return 1.0 if protected == 0.0 else float("inf")
    return protected / reference


def equalized_odds_difference(y_true, y_pred, group):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    tpr_gap = abs(
        _rate(y_pred, (group == 1) & (y_true == 1)) - _rate(y_pred, (group == 0) & (y_true == 1))
    )
    fpr_gap = abs(
        _rate(y_pred, (group == 1) & (y_true == 0)) - _rate(y_pred, (group == 0) & (y_true == 0))
    )
    return max(tpr_gap, fpr_gap)


def counterfactual_flip_rate(model, features, sensitive_index):
    flipped = np.array(features, dtype=float, copy=True)
    flipped[:, sensitive_index] = 1.0 - flipped[:, sensitive_index]
    return float(np.mean(model.predict(features) != model.predict(flipped)))


def attribution_entropy(phi):
    magnitude = np.abs(np.asarray(phi)).mean(axis=0)
    total = magnitude.sum()
    if total == 0.0:
        return 0.0
    share = magnitude / total
    share = share[share > 0]
    return float(-(share * np.log(share)).sum() / np.log(len(magnitude)))
