import argparse
import json
import sys
from pathlib import Path

import yaml

from .acs import load_public_coverage
from .data import frame_to_dataset, load_table, split_and_standardise, synthetic_public_benefit
from .ledger import AuditLedger, digest_array
from .report import annotate_report, review_status, write_report
from .trainer import train_with_accountability


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config):
    data_cfg = config["data"]
    source = data_cfg.get("source", "synthetic")
    if source == "synthetic":
        frame = synthetic_public_benefit(**data_cfg.get("synthetic", {}))
    elif source == "acs_public_coverage":
        frame = load_public_coverage(**data_cfg.get("acs", {}))
    else:
        frame = load_table(data_cfg["path"])
    dataset = frame_to_dataset(frame, data_cfg["target"], data_cfg["sensitive"], data_cfg.get("features"))
    return split_and_standardise(
        dataset, seed=data_cfg.get("split_seed", 7), standardise_binary=data_cfg.get("standardise_binary", False)
    )


def write_markdown_summary(path, config_path, result, verified):
    status = "Deployable, awaiting human approval" if result.deployable and verified else "Blocked"
    lines = [
        f"## EAA-AI audit: `{config_path}`",
        "",
        f"**Status:** {status}",
        "",
        f"Selected fairness weight λ: {result.selected_fairness_weight} (tried {', '.join(str(c['fairness_weight']) for c in result.candidates)})",
        "",
        f"Base rate: {result.diagnostics['base_rate']:.4f} | Constant predictor accuracy: {result.diagnostics['constant_predictor_accuracy']:.4f} | Reference model accuracy: {result.diagnostics['reference_model']['accuracy']:.4f} | Selected model accuracy: {result.diagnostics['selected_model']['accuracy']:.4f}",
        "",
        f"Selected iteration: {result.selected_iteration} | Rolled back: {'yes' if result.rolled_back else 'no'} | Ledger: {'verified' if verified else 'BROKEN'} | Head: `{result.ledger.head[:16]}`",
        "",
        "| Metric | Rule | Test value | 95% CI | Status |",
        "|---|---|---|---|---|",
    ]
    for name, spec in result.config["policy"].items():
        rule = f"{'≤' if spec['direction'] == 'max' else '≥'} {spec['threshold']}"
        passed = result.test_compliance["margins"][name] >= 0
        low, high = result.test_intervals[name]
        lines.append(f"| {name} | {rule} | {result.test_metrics[name]:.4f} | {low:.4f} to {high:.4f} | {'PASS' if passed else 'FAIL'} |")
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n\n")


def command_run(arguments):
    config = load_config(arguments.config)
    train, validation, test = build_dataset(config)
    result = train_with_accountability(train, validation, test, config)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = result.ledger.save(output / "audit_ledger.jsonl")
    verified, _ = AuditLedger.load(ledger_path).verify()
    report_path = write_report(result, test, output / "eaa_ai_audit_report.xlsx", verified)
    model_path = output / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "feature_names": result.feature_names,
                "weights": result.model.weights.tolist(),
                "bias": result.model.bias,
                "threshold": result.model.threshold,
                "parameters_hash": digest_array(result.model.parameters),
                "selected_iteration": result.selected_iteration,
                "selected_fairness_weight": result.selected_fairness_weight,
                "deployable": result.deployable,
                "ledger_head": result.ledger.head,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "deployable": result.deployable,
        "rolled_back": result.rolled_back,
        "selected_iteration": result.selected_iteration,
        "selected_fairness_weight": result.selected_fairness_weight,
        "violations": result.test_compliance["violations"],
        "ledger_verified": verified,
        "report": str(report_path),
    }
    print(json.dumps(summary, indent=2))
    if arguments.summary:
        write_markdown_summary(arguments.summary, arguments.config, result, verified)
    if arguments.strict and not (result.deployable and verified):
        return 2
    return 0


def command_verify(arguments):
    ledger = AuditLedger.load(arguments.ledger)
    verified, position = ledger.verify()
    print(json.dumps({"verified": verified, "records": len(ledger.records), "broken_at": position, "head": ledger.head}))
    return 0 if verified else 1


def command_approve(arguments):
    ledger = AuditLedger.load(arguments.ledger)
    verified, position = ledger.verify()
    if not verified:
        print(json.dumps({"error": "ledger integrity failure", "broken_at": position}))
        return 1
    decision = next((r for r in reversed(ledger.records) if r["event"] == "deployment_decision"), None)
    if decision is None:
        print(json.dumps({"error": "no deployment decision in ledger"}))
        return 1
    record = ledger.append(
        "human_review",
        {"role": arguments.reviewer_role, "unit": arguments.reviewer_unit},
        {
            "decision": arguments.decision,
            "note": arguments.note,
            "reviewed_record_hash": decision["hash"],
            "reviewed_parameters_hash": decision["payload"]["parameters_hash"],
        },
    )
    ledger.save(arguments.ledger)
    verified, _ = AuditLedger.load(arguments.ledger).verify()
    report = Path(arguments.ledger).with_name("eaa_ai_audit_report.xlsx")
    if report.exists():
        annotate_report(report, ledger, verified)
    if arguments.summary:
        write_review_summary(arguments.summary, ledger, record, decision, verified)
    print(json.dumps({"recorded": record["hash"], "decision": arguments.decision, "ledger_verified": verified, "report_updated": report.exists()}))
    return 0 if verified else 1


def write_review_summary(path, ledger, record, decision, verified):
    note = " ".join((record["payload"]["note"] or "None given").split()).replace("|", "\\|")
    lines = [
        "## EAA-AI human review",
        "",
        f"**Status:** {review_status(ledger)}",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Decision | {record['payload']['decision']} |",
        f"| Accountable role | {record['actor']['role']} |",
        f"| Reviewer | {record['actor']['unit']} |",
        f"| Justification | {note} |",
        f"| Recorded at | {record['timestamp'][:19].replace('T', ' ')} UTC |",
        f"| Model reviewed (parameters hash) | `{decision['payload']['parameters_hash'][:16]}` |",
        f"| Deployment decision reviewed | `{decision['hash'][:16]}` |",
        f"| Automated gate at review time | {'Deployable' if decision['payload']['deployable'] else 'Blocked'} |",
        f"| Ledger records | {len(ledger.records)} |",
        f"| Ledger integrity | {'Verified' if verified else 'BROKEN'} |",
        f"| Ledger head | `{ledger.head[:16]}` |",
    ]
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n\n")


def build_parser():
    parser = argparse.ArgumentParser(prog="eaa-ai")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--output", default="outputs")
    run.add_argument("--strict", action="store_true")
    run.add_argument("--summary", default=None)
    run.set_defaults(handler=command_run)
    verify = commands.add_parser("verify")
    verify.add_argument("--ledger", default="outputs/audit_ledger.jsonl")
    verify.set_defaults(handler=command_verify)
    approve = commands.add_parser("approve")
    approve.add_argument("--ledger", default="outputs/audit_ledger.jsonl")
    approve.add_argument("--reviewer-role", required=True)
    approve.add_argument("--reviewer-unit", default="oversight")
    approve.add_argument("--decision", choices=["approve", "reject"], required=True)
    approve.add_argument("--note", default="")
    approve.add_argument("--summary", default=None)
    approve.set_defaults(handler=command_approve)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
