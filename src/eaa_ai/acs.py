from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "public_coverage"
SENSITIVE = "protected_group"

NUMERIC = ["AGEP", "PINCP", "SCHL"]

BINARY = {
    "SEX": "male",
    "DIS": "disability",
    "DEAR": "hearing_difficulty",
    "DEYE": "vision_difficulty",
    "DREM": "cognitive_difficulty",
    "NATIVITY": "native_born",
}

CATEGORICAL = ["MAR", "ESP", "CIT", "MIG", "MIL", "ANC", "ESR", "ST", "FER"]

RAW_COLUMNS = NUMERIC + list(BINARY) + CATEGORICAL + ["RAC1P", "PUBCOV"]


def filter_population(raw):
    return raw[(raw["AGEP"] < 65) & (raw["PINCP"] <= 30000)]


def encode(raw, max_rows=None, seed=7):
    frame = filter_population(raw[RAW_COLUMNS].copy())
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=int(max_rows), random_state=seed)
    frame = frame.fillna(-1)
    encoded = pd.DataFrame(index=frame.index)
    for column in NUMERIC:
        encoded[column.lower()] = frame[column].astype(float)
    for column, name in BINARY.items():
        encoded[name] = (frame[column] == 1).astype(int)
    for column in CATEGORICAL:
        values = frame[column].astype(int)
        if values.nunique() < 2:
            continue
        dummies = pd.get_dummies(values, prefix=column, dtype=int)
        encoded = pd.concat([encoded, dummies], axis=1)
    encoded[SENSITIVE] = (frame["RAC1P"] != 1).astype(int)
    encoded[TARGET] = (frame["PUBCOV"] == 1).astype(int)
    return encoded.reset_index(drop=True)


def download(states, year, horizon, root):
    from folktables import ACSDataSource

    Path(root).mkdir(parents=True, exist_ok=True)
    source = ACSDataSource(survey_year=str(year), horizon=horizon, survey="person", root_dir=str(root))
    return source.get_data(states=list(states), download=True)


def load_public_coverage(states=("CA",), year=2018, horizon="1-Year", root=".cache/acs", max_rows=None, seed=7):
    raw = download(states, year, horizon, root)
    return encode(raw, max_rows=max_rows, seed=seed)

