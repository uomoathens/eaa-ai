from dataclasses import dataclass, field

import numpy as np

from .compliance import CompliancePolicy
from .explain import LinearShapExplainer
from .ledger import AuditLedger, digest_array, digest_payload
from .metrics import (
    accuracy,
    attribution_entropy,
    counterfactual_flip_rate,
    disparate_impact_ratio,
    equalized_odds_difference,
    statistical_parity_difference,
)
from .model import FairLogisticRegression
from .reward import responsible_reward_original, responsible_reward_refined


@dataclass
class RunResult:
    model: FairLogisticRegression
    ledger: AuditLedger
    history: list
    selected_iteration: int
    deployable: bool
    test_metrics: dict
    test_compliance: dict
    explanation: object
    feature_names: list
    rolled_back: bool
    config: dict = field(default_factory=dict)


def evaluate(model, split, explainer, proxy_indices):
    predictions = model.predict(split.features)
    explanation = explainer.explain(model, split.features, split.feature_names)
    watched = [split.sensitive_index, *proxy_indices]
    return {
        "accuracy": accuracy(split.labels, predictions),
        "abs_statistical_parity_difference": abs(statistical_parity_difference(predictions, split.group)),
        "disparate_impact_ratio": disparate_impact_ratio(predictions, split.group),
        "equalized_odds_difference": equalized_odds_difference(split.labels, predictions, split.group),
        "counterfactual_flip_rate": counterfactual_flip_rate(model, split.features, split.sensitive_index),
        "sensitive_attribution_share": explanation.share_of(watched),
        "attribution_concentration": 1.0 - attribution_entropy(explanation.phi),
    }, explanation


def train_with_accountability(train, validation, test, config):
    model_cfg = config["model"]
    reward_cfg = config["reward"]
    actor = config["actor"]
    policy = CompliancePolicy.from_mapping(config["policy"])
    proxy_indices = [train.feature_names.index(name) for name in config.get("proxy_features", [])]
    explainer = LinearShapExplainer(train.features)
    model = FairLogisticRegression(
        n_features=train.features.shape[1],
        fairness_weight=model_cfg["fairness_weight"],
        explain_weight=model_cfg["explain_weight"],
        learning_rate=model_cfg["learning_rate"],
        threshold=model_cfg.get("threshold", 0.5),
        blind_indices=[train.feature_names.index(name) for name in model_cfg.get("blind_features", [])],
    )
    ledger = AuditLedger()
    ledger.append(
        "run_started",
        actor,
        {
            "config_hash": digest_payload(config),
            "train_fingerprint": train.fingerprint(),
            "validation_fingerprint": validation.fingerprint(),
            "test_fingerprint": test.fingerprint(),
            "feature_names": train.feature_names,
            "initial_parameters_hash": digest_array(model.parameters),
        },
    )
    history = []
    last_compliant = None
    for iteration in range(1, int(model_cfg["iterations"]) + 1):
        delta = model.step(train.features, train.labels, train.group)
        losses = model.loss_terms(train.features, train.labels, train.group)
        metrics, _ = evaluate(model, validation, explainer, proxy_indices)
        compliance = policy.evaluate(metrics)
        rewards = {
            "original": responsible_reward_original(
                metrics["accuracy"],
                metrics["abs_statistical_parity_difference"],
                compliance["c_ethics"],
                reward_cfg["gamma_bias"],
                reward_cfg["gamma_ethics"],
            ),
            "refined": responsible_reward_refined(
                metrics["accuracy"],
                metrics["abs_statistical_parity_difference"],
                compliance["hinge_penalty"],
                reward_cfg["gamma_bias"],
                reward_cfg["gamma_ethics"],
            ),
        }
        record = ledger.append(
            "parameter_update",
            actor,
            {
                "iteration": iteration,
                "delta_hash": digest_array(delta),
                "delta_norm": float(np.linalg.norm(delta)),
                "parameters_hash": digest_array(model.parameters),
                "losses": losses,
                "metrics": metrics,
                "compliance": compliance,
                "rewards": rewards,
                "flagged": not compliance["deployable"],
            },
        )
        if compliance["deployable"]:
            last_compliant = (iteration, model.parameters.copy(), record["hash"])
        history.append({"iteration": iteration, **losses, **metrics, **rewards, **compliance})
    rolled_back = False
    final_iteration = int(model_cfg["iterations"])
    if history and not history[-1]["deployable"] and last_compliant is not None:
        rolled_back = True
        final_iteration = last_compliant[0]
        model.set_parameters(last_compliant[1])
        ledger.append(
            "rollback",
            actor,
            {
                "from_iteration": int(model_cfg["iterations"]),
                "to_iteration": last_compliant[0],
                "restored_record_hash": last_compliant[2],
                "parameters_hash": digest_array(model.parameters),
            },
        )
    test_metrics, explanation = evaluate(model, test, explainer, proxy_indices)
    test_compliance = policy.evaluate(test_metrics)
    ledger.append(
        "deployment_decision",
        actor,
        {
            "selected_iteration": final_iteration,
            "parameters_hash": digest_array(model.parameters),
            "test_metrics": test_metrics,
            "test_compliance": test_compliance,
            "deployable": test_compliance["deployable"],
            "requires_human_approval": True,
            "global_importance": explanation.global_importance(),
        },
    )
    return RunResult(
        model=model,
        ledger=ledger,
        history=history,
        selected_iteration=final_iteration,
        deployable=test_compliance["deployable"],
        test_metrics=test_metrics,
        test_compliance=test_compliance,
        explanation=explanation,
        feature_names=train.feature_names,
        rolled_back=rolled_back,
        config=config,
    )
