import numpy as np

from .metrics import (
    attribution_entropy,
    disparate_impact_ratio,
    equalized_odds_difference,
    statistical_parity_difference,
)


def _flip(features, sensitive_index):
    flipped = np.array(features, dtype=float, copy=True)
    flipped[:, sensitive_index] = 1.0 - flipped[:, sensitive_index]
    return flipped


def _metrics(labels, group, predictions, flipped, magnitude, watched):
    base_rate = float(np.mean(labels))
    acc = float(np.mean(predictions == labels))
    mean_magnitude = magnitude.mean(axis=0)
    total = mean_magnitude.sum()
    return {
        "accuracy": acc,
        "accuracy_lift": acc - max(base_rate, 1.0 - base_rate),
        "abs_statistical_parity_difference": abs(statistical_parity_difference(predictions, group)),
        "disparate_impact_ratio": disparate_impact_ratio(predictions, group),
        "equalized_odds_difference": equalized_odds_difference(labels, predictions, group),
        "counterfactual_flip_rate": float(np.mean(predictions != flipped)),
        "sensitive_attribution_share": float(mean_magnitude[watched].sum() / total) if total else 0.0,
        "attribution_concentration": 1.0 - attribution_entropy(magnitude),
    }


def bootstrap_metrics(model, split, explainer, proxy_indices, samples=200, seed=0):
    predictions = model.predict(split.features)
    flipped = model.predict(_flip(split.features, split.sensitive_index))
    magnitude = np.abs(explainer.explain(model, split.features, split.feature_names).phi)
    watched = [split.sensitive_index, *proxy_indices]
    rng = np.random.default_rng(seed)
    n = len(split.labels)
    draws = []
    for _ in range(int(samples)):
        index = rng.integers(0, n, n)
        draws.append(
            _metrics(split.labels[index], split.group[index], predictions[index], flipped[index], magnitude[index], watched)
        )
    return {name: np.array([draw[name] for draw in draws]) for name in draws[0]}


def confidence_margins(policy, samples, confidence):
    alpha = 1.0 - float(confidence)
    margins = {}
    for requirement in policy.requirements:
        values = samples[requirement.metric]
        if requirement.direction == "min":
            margins[requirement.metric] = float(np.quantile(values, alpha)) - requirement.threshold
        else:
            margins[requirement.metric] = requirement.threshold - float(np.quantile(values, 1.0 - alpha))
    return margins


def intervals(samples, confidence=0.95):
    alpha = (1.0 - float(confidence)) / 2.0
    return {name: (float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))) for name, values in samples.items()}
