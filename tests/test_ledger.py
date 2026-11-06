import numpy as np

from eaa_ai.ledger import GENESIS, AuditLedger, digest_array

ACTOR = {"role": "provider", "unit": "ml-engineering"}


def build_ledger():
    ledger = AuditLedger()
    for step in range(5):
        ledger.append("parameter_update", ACTOR, {"iteration": step, "delta_hash": digest_array(np.ones(3) * step)})
    return ledger


def test_chain_links_and_verifies():
    ledger = build_ledger()
    assert ledger.records[0]["previous_hash"] == GENESIS
    for previous, current in zip(ledger.records, ledger.records[1:]):
        assert current["previous_hash"] == previous["hash"]
    assert ledger.verify() == (True, None)


def test_payload_tampering_is_detected():
    ledger = build_ledger()
    ledger.records[2]["payload"]["iteration"] = 99
    assert ledger.verify() == (False, 2)


def test_record_deletion_is_detected():
    ledger = build_ledger()
    del ledger.records[1]
    verified, position = ledger.verify()
    assert verified is False
    assert position == 1


def test_round_trip_preserves_integrity(tmp_path):
    ledger = build_ledger()
    path = ledger.save(tmp_path / "ledger.jsonl")
    restored = AuditLedger.load(path)
    assert restored.head == ledger.head
    assert restored.verify() == (True, None)
