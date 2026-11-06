import pytest

from eaa_ai.compliance import CompliancePolicy
from eaa_ai.reward import responsible_reward_original, responsible_reward_refined

POLICY = CompliancePolicy.from_mapping(
    {
        "accuracy": {"threshold": 0.75, "direction": "min"},
        "abs_statistical_parity_difference": {"threshold": 0.10, "direction": "max"},
    }
)


def test_compliant_metrics_pass_the_gate():
    result = POLICY.evaluate({"accuracy": 0.9, "abs_statistical_parity_difference": 0.05})
    assert result["c_ethics"] == 1
    assert result["hinge_penalty"] == 0.0
    assert result["deployable"] is True
    assert result["violations"] == []


def test_hinge_penalty_measures_shortfall():
    result = POLICY.evaluate({"accuracy": 0.70, "abs_statistical_parity_difference": 0.13})
    assert result["c_ethics"] == 0
    assert result["hinge_penalty"] == pytest.approx(0.05 + 0.03)
    assert result["violations"] == ["abs_statistical_parity_difference", "accuracy"]


def test_original_reward_is_blind_to_size_of_violation():
    small = POLICY.evaluate({"accuracy": 0.9, "abs_statistical_parity_difference": 0.101})
    large = POLICY.evaluate({"accuracy": 0.9, "abs_statistical_parity_difference": 0.40})
    original_small = responsible_reward_original(0.9, 0.1, small["c_ethics"], 1.0, 0.5)
    original_large = responsible_reward_original(0.9, 0.1, large["c_ethics"], 1.0, 0.5)
    refined_small = responsible_reward_refined(0.9, 0.1, small["hinge_penalty"], 1.0, 0.5)
    refined_large = responsible_reward_refined(0.9, 0.1, large["hinge_penalty"], 1.0, 0.5)
    assert original_small == original_large
    assert refined_small > refined_large


def test_unknown_direction_is_rejected():
    policy = CompliancePolicy.from_mapping({"accuracy": {"threshold": 0.5, "direction": "around"}})
    with pytest.raises(ValueError):
        policy.evaluate({"accuracy": 0.6})
