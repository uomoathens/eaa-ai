import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GENESIS = "0" * 64


def digest_array(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def digest_payload(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class AuditLedger:
    records: list = field(default_factory=list)

    @property
    def head(self):
        return self.records[-1]["hash"] if self.records else GENESIS

    def append(self, event, actor, payload):
        body = {
            "index": len(self.records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "actor": dict(actor),
            "payload": payload,
            "previous_hash": self.head,
        }
        body["hash"] = digest_payload(body)
        self.records.append(body)
        return body

    def verify(self):
        previous = GENESIS
        for position, record in enumerate(self.records):
            body = {key: value for key, value in record.items() if key != "hash"}
            if record["index"] != position:
                return False, position
            if record["previous_hash"] != previous:
                return False, position
            if digest_payload(body) != record["hash"]:
                return False, position
            previous = record["hash"]
        return True, None

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return target

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return cls(records=records)
