import json
from pathlib import Path

from openpyxl import load_workbook

from eaa_ai.cli import main
from eaa_ai.ledger import AuditLedger

ROOT = Path(__file__).resolve().parents[1]


def test_default_run_is_deployable_and_reported(tmp_path, capsys):
    code = main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--output", str(tmp_path), "--strict"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["deployable"] is True
    assert summary["ledger_verified"] is True
    workbook = load_workbook(tmp_path / "eaa_ai_audit_report.xlsx")
    assert workbook.sheetnames == [
        "Summary",
        "Requirements",
        "Training history",
        "Audit ledger",
        "Global explanation",
        "Case explanations",
    ]
    model = json.loads((tmp_path / "model.json").read_text())
    assert model["ledger_head"] == AuditLedger.load(tmp_path / "audit_ledger.jsonl").head


def test_baseline_without_fairness_fails_strict_gate(tmp_path, capsys):
    code = main(["run", "--config", str(ROOT / "configs" / "baseline.yaml"), "--output", str(tmp_path), "--strict"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 2
    assert summary["deployable"] is False
    assert summary["violations"]


def test_human_review_extends_a_valid_chain(tmp_path, capsys):
    main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--output", str(tmp_path)])
    ledger_path = tmp_path / "audit_ledger.jsonl"
    code = main(["approve", "--ledger", str(ledger_path), "--reviewer-role", "deployer", "--decision", "approve"])
    capsys.readouterr()
    ledger = AuditLedger.load(ledger_path)
    assert code == 0
    assert ledger.records[-1]["event"] == "human_review"
    assert ledger.verify() == (True, None)
    assert main(["verify", "--ledger", str(ledger_path)]) == 0
