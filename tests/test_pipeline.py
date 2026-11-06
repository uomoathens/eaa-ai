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
        "Model selection",
        "Diagnostics",
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
    capsys.readouterr()
    ledger_path = tmp_path / "audit_ledger.jsonl"
    summary_path = tmp_path / "summary.md"
    code = main(
        [
            "approve",
            "--ledger",
            str(ledger_path),
            "--reviewer-role",
            "deployer",
            "--reviewer-unit",
            "reviewer-account",
            "--decision",
            "approve",
            "--note",
            "Checked | thresholds\nand intervals",
            "--summary",
            str(summary_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    ledger = AuditLedger.load(ledger_path)
    assert code == 0
    assert output["report_updated"] is True
    assert ledger.records[-1]["event"] == "human_review"
    assert ledger.verify() == (True, None)
    assert main(["verify", "--ledger", str(ledger_path)]) == 0
    workbook = load_workbook(tmp_path / "eaa_ai_audit_report.xlsx")
    assert workbook.sheetnames[1] == "Human review"
    review = workbook["Human review"]
    assert review.max_row == 2
    assert review.cell(row=2, column=2).value == "approve"
    assert workbook["Audit ledger"].max_row == len(ledger.records) + 1
    summary = {row[0].value: row[1].value for row in workbook["Summary"].iter_rows(min_row=2)}
    assert summary["Human approval"].startswith("Approved by deployer (reviewer-account)")
    assert summary["Ledger records"] == len(ledger.records)
    assert summary["Ledger head hash"] == ledger.head
    text = summary_path.read_text(encoding="utf-8")
    assert "Approved by deployer" in text
    assert "Checked \\| thresholds and intervals" in text


def test_review_is_refused_on_a_tampered_ledger(tmp_path, capsys):
    main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--output", str(tmp_path)])
    ledger_path = tmp_path / "audit_ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[3])
    record["payload"]["flagged"] = not record["payload"]["flagged"]
    lines[3] = json.dumps(record, sort_keys=True)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = main(
        ["approve", "--ledger", str(ledger_path), "--reviewer-role", "deployer", "--decision", "approve", "--note", "Checked"]
    )
    capsys.readouterr()
    assert code == 1
    assert AuditLedger.load(ledger_path).records[-1]["event"] != "human_review"


def approve(ledger_path, note):
    return main(["approve", "--ledger", str(ledger_path), "--reviewer-role", "deployer", "--decision", "approve", "--note", note])


def test_review_without_justification_is_refused(tmp_path, capsys):
    main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--output", str(tmp_path)])
    ledger_path = tmp_path / "audit_ledger.jsonl"
    before = len(AuditLedger.load(ledger_path).records)
    assert approve(ledger_path, "   ") == 1
    capsys.readouterr()
    assert len(AuditLedger.load(ledger_path).records) == before


def test_successive_reviews_accumulate_and_extend_the_audit(tmp_path, capsys):
    main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--output", str(tmp_path)])
    ledger_path = tmp_path / "audit_ledger.jsonl"
    base_path = tmp_path / "base.jsonl"
    base_path.write_text(ledger_path.read_text(encoding="utf-8"), encoding="utf-8")
    base_count = len(AuditLedger.load(base_path).records)
    assert approve(ledger_path, "First review") == 0
    assert approve(ledger_path, "Second review") == 0
    capsys.readouterr()
    ledger = AuditLedger.load(ledger_path)
    assert len(ledger.records) == base_count + 2
    assert [r["payload"]["note"] for r in ledger.records[-2:]] == ["First review", "Second review"]
    assert main(["verify", "--ledger", str(ledger_path), "--extends", str(base_path)]) == 0
    assert json.loads(capsys.readouterr().out)["extends_base"] is True
    review = load_workbook(tmp_path / "eaa_ai_audit_report.xlsx")["Human review"]
    assert review.max_row == 3
    other = tmp_path / "other"
    main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--output", str(other)])
    capsys.readouterr()
    assert main(["verify", "--ledger", str(ledger_path), "--extends", str(other / "audit_ledger.jsonl")]) == 1
    assert json.loads(capsys.readouterr().out)["extends_base"] is False
