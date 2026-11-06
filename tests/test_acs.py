import json
from pathlib import Path

import yaml

from eaa_ai import acs
from eaa_ai.cli import main
from eaa_ai.ledger import AuditLedger

ROOT = Path(__file__).resolve().parents[1]


def test_encoding_follows_folktables_task_definition(acs_raw):
    encoded = acs.encode(acs_raw)
    expected = acs_raw[(acs_raw["AGEP"] < 65) & (acs_raw["PINCP"] <= 30000)]
    assert len(encoded) == len(expected)
    assert encoded[acs.TARGET].sum() == int((expected["PUBCOV"] == 1).sum())
    assert encoded[acs.SENSITIVE].sum() == int((expected["RAC1P"] != 1).sum())
    assert not encoded.isna().any().any()
    assert not any(column.startswith("ST_") for column in encoded.columns)
    assert "RAC1P" not in encoded.columns
    assert {"CIT_1", "ANC_1", "native_born", "disability"}.issubset(encoded.columns)


def test_row_cap_is_reproducible(acs_raw):
    first = acs.encode(acs_raw, max_rows=500, seed=3)
    second = acs.encode(acs_raw, max_rows=500, seed=3)
    assert len(first) == 500
    assert first.equals(second)


def test_acs_configuration_runs_end_to_end(acs_raw, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(acs, "download", lambda states, year, horizon, root: acs_raw)
    config = yaml.safe_load((ROOT / "configs" / "acs_public_coverage.yaml").read_text())
    config["model"]["iterations"] = 60
    config_path = tmp_path / "acs.yaml"
    config_path.write_text(yaml.safe_dump(config))
    code = main(["run", "--config", str(config_path), "--output", str(tmp_path / "out")])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0
    assert summary["ledger_verified"] is True
    ledger = AuditLedger.load(tmp_path / "out" / "audit_ledger.jsonl")
    decision = ledger.records[-1]["payload"]
    assert ledger.records[-1]["event"] == "deployment_decision"
    assert set(decision["test_metrics"]) >= set(config["policy"])
    model = json.loads((tmp_path / "out" / "model.json").read_text())
    assert model["weights"][model["feature_names"].index("protected_group")] == 0.0
