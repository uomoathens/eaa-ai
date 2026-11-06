import argparse
import json
import sys
from pathlib import Path

import yaml

from .data import frame_to_dataset, load_table, split_and_standardise, synthetic_public_benefit
from .ledger import AuditLedger, digest_array
from .report import write_report
from .trainer import train_with_accountability


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config):
    data_cfg = config["data"]
    if data_cfg.get("source", "synthetic") == "synthetic":
        frame = synthetic_public_benefit(**data_cfg.get("synthetic", {}))
    else:
        frame = load_table(data_cfg["path"])
    dataset = frame_to_dataset(frame, data_cfg["target"], data_cfg["sensitive"], data_cfg.get("features"))
    return split_and_standardise(dataset, seed=data_cfg.get("split_seed", 7))


def write_markdown_summary(path, config_path, result, verified):
    status = "Deployable, awaiting human approval" if result.deployable and verified else "Blocked"
    lines = [
        f"## EAA-AI audit: `{config_path}`",
        "",
        f"**Status:** {status}",
        "",
        f"Selected iteration: {result.selected_iteration} | Rolled back: {'yes' if result.rolled_back else 'no'} | Ledger: {'verified' if verified else 'BROKEN'} | Head: `{result.ledger.head[:16]}`",
        "",
        "| Metric | Rule | Test value | Status |",
        "|---|---|---|---|",
    ]
    for name, spec in result.config["policy"].items():
        rule = f"{'≤' if spec['direction'] == 'max' else '≥'} {spec['threshold']}"
        passed = result.test_compliance["margins"][name] >= 0
        lines.append(f"| {name} | {rule} | {result.test_metrics[name]:.4f} | {'PASS' if passed else 'FAIL'} |")
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
    print(json.dumps({"recorded": record["hash"], "decision": arguments.decision}))
    return 0


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
    approve.set_defaults(handler=command_approve)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
