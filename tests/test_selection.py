import numpy as np
import pytest

from eaa_ai.bootstrap import bootstrap_metrics, confidence_margins, intervals
from eaa_ai.compliance import CompliancePolicy
from eaa_ai.data import frame_to_dataset, split_and_standardise, synthetic_public_benefit
from eaa_ai.explain import LinearShapExplainer
from eaa_ai.model import FairLogisticRegression
from eaa_ai.trainer import _select, evaluate


@pytest.fixture(scope="module")
def splits():
    frame = synthetic_public_benefit(n_samples=4000, historical_bias=0.6, seed=5)
    return split_and_standardise(frame_to_dataset(frame, "eligible", "protected_group"), seed=5, standardise_binary=True)


def fitted(train):
    model = FairLogisticRegression(train.features.shape[1], learning_rate=0.3, blind_indices=[train.sensitive_index])
    for _ in range(200):
        model.step(train.features, train.labels, train.group)
    return model


def test_sensitive_column_is_never_standardised(splits):
    train, validation, _ = splits
    assert set(np.unique(train.features[:, train.sensitive_index])) == {0.0, 1.0}
    assert set(np.unique(validation.features[:, validation.sensitive_index])) == {0.0, 1.0}


def test_accuracy_lift_is_relative_to_constant_predictor(splits):
    train, validation, _ = splits
    model = fitted(train)
    metrics, _ = evaluate(model, validation, LinearShapExplainer(train.features), [])
    base = validation.labels.mean()
    assert metrics["accuracy_lift"] == pytest.approx(metrics["accuracy"] - max(base, 1 - base))


def test_bootstrap_intervals_bracket_point_estimates(splits):
    train, validation, _ = splits
    model = fitted(train)
    explainer = LinearShapExplainer(train.features)
    point, _ = evaluate(model, validation, explainer, [])
    bounds = intervals(bootstrap_metrics(model, validation, explainer, [], samples=300, seed=1))
    for name in ("accuracy", "disparate_impact_ratio", "abs_statistical_parity_difference"):
        low, high = bounds[name]
        assert low <= point[name] <= high
        assert high - low > 0


def test_confidence_margin_is_stricter_than_point_margin(splits):
    train, validation, _ = splits
    model = fitted(train)
    explainer = LinearShapExplainer(train.features)
    point, _ = evaluate(model, validation, explainer, [])
    policy = CompliancePolicy.from_mapping({"disparate_impact_ratio": {"threshold": 0.8, "direction": "min"}})
    samples = bootstrap_metrics(model, validation, explainer, [], samples=300, seed=1)
    assert confidence_margins(policy, samples, 0.95)["disparate_impact_ratio"] < policy.margins(point)["disparate_impact_ratio"]


def candidate(weight, confident, worst):
    return {"fairness_weight": weight, "passes_with_confidence": confident, "confidence_margins": {"m": worst}}


def test_selection_prefers_smallest_confident_weight():
    chosen, reason = _select([candidate(0.0, False, -0.1), candidate(3.0, True, 0.01), candidate(10.0, True, 0.05)], 0.95)
    assert chosen["fairness_weight"] == 3.0
    assert "95%" in reason


def test_selection_falls_back_to_closest_candidate():
    chosen, reason = _select([candidate(0.0, False, -0.1), candidate(3.0, False, -0.02), candidate(10.0, False, -0.05)], 0.95)
    assert chosen["fairness_weight"] == 3.0
    assert "closest" in reason
