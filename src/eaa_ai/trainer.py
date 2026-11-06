from dataclasses import dataclass, field

import numpy as np

from .bootstrap import bootstrap_metrics, confidence_margins, intervals
from .compliance import CompliancePolicy
from .diagnostics import run_diagnostics
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


def resolve_features(feature_names, patterns):
    indices = []
    for pattern in patterns:
        matches = [i for i, name in enumerate(feature_names) if name == pattern or name.startswith(f"{pattern}_")]
        if not matches:
            raise KeyError(f"no feature matches {pattern}")
        indices.extend(matches)
    return sorted(set(indices))


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
    selected_fairness_weight: float = 0.0
    candidates: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    test_intervals: dict = field(default_factory=dict)


def evaluate(model, split, explainer, proxy_indices):
    predictions = model.predict(split.features)
    explanation = explainer.explain(model, split.features, split.feature_names)
    watched = [split.sensitive_index, *proxy_indices]
    base_rate = float(np.mean(split.labels))
    acc = accuracy(split.labels, predictions)
    return {
        "accuracy": acc,
        "accuracy_lift": acc - max(base_rate, 1.0 - base_rate),
        "abs_statistical_parity_difference": abs(statistical_parity_difference(predictions, split.group)),
        "disparate_impact_ratio": disparate_impact_ratio(predictions, split.group),
        "equalized_odds_difference": equalized_odds_difference(split.labels, predictions, split.group),
        "counterfactual_flip_rate": counterfactual_flip_rate(model, split.features, split.sensitive_index),
        "sensitive_attribution_share": explanation.share_of(watched),
        "attribution_concentration": 1.0 - attribution_entropy(explanation.phi),
    }, explanation


def _rewards(metrics, compliance, reward_cfg):
    return {
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


def _train_candidate(fairness_weight, train, validation, context):
    model_cfg = context["model_cfg"]
    ledger = context["ledger"]
    actor = context["actor"]
    policy = context["policy"]
    iterations = int(model_cfg["iterations"])
    model = FairLogisticRegression(
        n_features=train.features.shape[1],
        fairness_weight=fairness_weight,
        explain_weight=model_cfg["explain_weight"],
        learning_rate=model_cfg["learning_rate"],
        threshold=model_cfg.get("threshold", 0.5),
        blind_indices=context["blind_indices"],
    )
    history = []
    last_compliant = None
    for iteration in range(1, iterations + 1):
        delta = model.step(train.features, train.labels, train.group)
        losses = model.loss_terms(train.features, train.labels, train.group)
        metrics, _ = evaluate(model, validation, context["explainer"], context["proxy_indices"])
        compliance = policy.evaluate(metrics)
        rewards = _rewards(metrics, compliance, context["reward_cfg"])
        record = ledger.append(
            "parameter_update",
            actor,
            {
                "fairness_weight": fairness_weight,
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
        history.append({"fairness_weight": fairness_weight, "iteration": iteration, **losses, **metrics, **rewards, **compliance})
    selected_iteration = iterations
    rolled_back = False
    if not history[-1]["deployable"] and last_compliant is not None:
        rolled_back = True
        selected_iteration = last_compliant[0]
        model.set_parameters(last_compliant[1])
        ledger.append(
            "rollback",
            actor,
            {
                "fairness_weight": fairness_weight,
                "from_iteration": iterations,
                "to_iteration": last_compliant[0],
                "restored_record_hash": last_compliant[2],
                "parameters_hash": digest_array(model.parameters),
            },
        )
    metrics, _ = evaluate(model, validation, context["explainer"], context["proxy_indices"])
    compliance = policy.evaluate(metrics)
    samples = bootstrap_metrics(
        model, validation, context["explainer"], context["proxy_indices"], context["bootstrap_samples"], context["seed"]
    )
    lower_margins = confidence_margins(policy, samples, context["confidence"])
    confident = min(lower_margins.values()) >= 0.0
    ledger.append(
        "candidate_evaluated",
        actor,
        {
            "fairness_weight": fairness_weight,
            "selected_iteration": selected_iteration,
            "rolled_back": rolled_back,
            "parameters_hash": digest_array(model.parameters),
            "validation_metrics": metrics,
            "validation_compliance": compliance,
            "confidence": context["confidence"],
            "confidence_margins": lower_margins,
            "passes_with_confidence": confident,
        },
    )
    return {
        "confidence_margins": lower_margins,
        "passes_with_confidence": confident,
        "fairness_weight": fairness_weight,
        "model": model,
        "history": history,
        "selected_iteration": selected_iteration,
        "rolled_back": rolled_back,
        "validation_metrics": metrics,
        "validation_compliance": compliance,
    }


def _select(candidates, confidence):
    compliant = [c for c in candidates if c["passes_with_confidence"]]
    if compliant:
        return min(compliant, key=lambda c: c["fairness_weight"]), f"smallest fairness weight passing every requirement on validation with {confidence:.0%} confidence"
    best = max(candidates, key=lambda c: (min(c["confidence_margins"].values()), -c["fairness_weight"]))
    return best, f"no candidate passed on validation with {confidence:.0%} confidence; candidate closest to passing selected"


def train_with_accountability(train, validation, test, config):
    model_cfg = config["model"]
    actor = config["actor"]
    grid = sorted(float(value) for value in model_cfg.get("fairness_weight_grid", [model_cfg["fairness_weight"]]))
    blind_indices = resolve_features(train.feature_names, model_cfg.get("blind_features", []))
    context = {
        "model_cfg": model_cfg,
        "reward_cfg": config["reward"],
        "actor": actor,
        "policy": CompliancePolicy.from_mapping(config["policy"]),
        "proxy_indices": resolve_features(train.feature_names, config.get("proxy_features", [])),
        "blind_indices": blind_indices,
        "explainer": LinearShapExplainer(train.features),
        "ledger": AuditLedger(),
        "confidence": float(model_cfg.get("selection_confidence", 0.95)),
        "bootstrap_samples": int(model_cfg.get("bootstrap_samples", 200)),
        "seed": int(config["data"].get("split_seed", 7)),
    }
    ledger = context["ledger"]
    ledger.append(
        "run_started",
        actor,
        {
            "config_hash": digest_payload(config),
            "train_fingerprint": train.fingerprint(),
            "validation_fingerprint": validation.fingerprint(),
            "test_fingerprint": test.fingerprint(),
            "feature_names": train.feature_names,
            "fairness_weight_grid": grid,
        },
    )
    candidates = [_train_candidate(weight, train, validation, context) for weight in grid]
    chosen, reason = _select(candidates, context["confidence"])
    model = chosen["model"]
    ledger.append(
        "model_selection",
        actor,
        {
            "selected_fairness_weight": chosen["fairness_weight"],
            "reason": reason,
            "parameters_hash": digest_array(model.parameters),
            "candidates": [
                {
                    "fairness_weight": c["fairness_weight"],
                    "deployable_on_validation": c["validation_compliance"]["deployable"],
                    "hinge_penalty": c["validation_compliance"]["hinge_penalty"],
                    "passes_with_confidence": c["passes_with_confidence"],
                    "worst_confidence_margin": min(c["confidence_margins"].values()),
                }
                for c in candidates
            ],
        },
    )
    test_metrics, explanation = evaluate(model, test, context["explainer"], context["proxy_indices"])
    test_compliance = context["policy"].evaluate(test_metrics)
    test_intervals = intervals(
        bootstrap_metrics(model, test, context["explainer"], context["proxy_indices"], context["bootstrap_samples"], context["seed"] + 1)
    )
    diagnostics = run_diagnostics(model, train, test, blind_indices)
    ledger.append(
        "deployment_decision",
        actor,
        {
            "selected_fairness_weight": chosen["fairness_weight"],
            "selected_iteration": chosen["selected_iteration"],
            "parameters_hash": digest_array(model.parameters),
            "test_metrics": test_metrics,
            "test_compliance": test_compliance,
            "test_intervals_95": test_intervals,
            "deployable": test_compliance["deployable"],
            "requires_human_approval": True,
            "global_importance": explanation.global_importance(),
            "diagnostics": diagnostics,
        },
    )
    history = [row for c in candidates for row in c["history"]]
    return RunResult(
        model=model,
        ledger=ledger,
        history=history,
        selected_iteration=chosen["selected_iteration"],
        deployable=test_compliance["deployable"],
        test_metrics=test_metrics,
        test_compliance=test_compliance,
        explanation=explanation,
        feature_names=train.feature_names,
        rolled_back=chosen["rolled_back"],
        config=config,
        selected_fairness_weight=chosen["fairness_weight"],
        candidates=[
            {
                "fairness_weight": c["fairness_weight"],
                "selected_iteration": c["selected_iteration"],
                "rolled_back": c["rolled_back"],
                "deployable_on_validation": c["validation_compliance"]["deployable"],
                "hinge_penalty": c["validation_compliance"]["hinge_penalty"],
                "passes_with_confidence": c["passes_with_confidence"],
                "worst_confidence_margin": min(c["confidence_margins"].values()),
                "violations": ", ".join(c["validation_compliance"]["violations"]) or "No violations",
                **{f"val_{k}": v for k, v in c["validation_metrics"].items()},
            }
            for c in candidates
        ],
        diagnostics=diagnostics,
        test_intervals=test_intervals,
    )
