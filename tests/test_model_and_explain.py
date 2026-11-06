import numpy as np
import pytest

from eaa_ai.data import frame_to_dataset, split_and_standardise, synthetic_public_benefit
from eaa_ai.explain import LinearShapExplainer
from eaa_ai.metrics import (
    counterfactual_flip_rate,
    disparate_impact_ratio,
    equalized_odds_difference,
    statistical_parity_difference,
)
from eaa_ai.model import FairLogisticRegression


@pytest.fixture(scope="module")
def splits():
    frame = synthetic_public_benefit(n_samples=3000, historical_bias=0.6, seed=3)
    dataset = frame_to_dataset(frame, "eligible", "protected_group")
    return split_and_standardise(dataset, seed=3)


def fit(train, fairness_weight, blind=()):
    model = FairLogisticRegression(
        train.features.shape[1], fairness_weight=fairness_weight, learning_rate=0.2, blind_indices=blind
    )
    for _ in range(300):
        model.step(train.features, train.labels, train.group)
    return model


def test_fairness_penalty_reduces_parity_gap(splits):
    train, validation, _ = splits
    baseline = fit(train, 0.0)
    fair = fit(train, 40.0)
    gap_baseline = abs(statistical_parity_difference(baseline.predict(validation.features), validation.group))
    gap_fair = abs(statistical_parity_difference(fair.predict(validation.features), validation.group))
    assert gap_fair < gap_baseline


def test_blind_feature_weight_stays_zero(splits):
    train, validation, _ = splits
    model = fit(train, 10.0, blind=[train.sensitive_index])
    assert model.weights[train.sensitive_index] == 0.0
    assert counterfactual_flip_rate(model, validation.features, validation.sensitive_index) == 0.0


def test_shap_values_are_locally_accurate(splits):
    train, validation, _ = splits
    model = fit(train, 5.0)
    explanation = LinearShapExplainer(train.features).explain(model, validation.features, validation.feature_names)
    assert np.allclose(explanation.reconstruct(), model.decision_function(validation.features))
    assert len(explanation.top_k(0, 3)) == 3


def test_metric_values_on_known_vectors():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert statistical_parity_difference(y_pred, group) == pytest.approx(-0.25)
    assert disparate_impact_ratio(y_pred, group) == pytest.approx(0.5)
    assert equalized_odds_difference(y_true, y_pred, group) == pytest.approx(0.5)
