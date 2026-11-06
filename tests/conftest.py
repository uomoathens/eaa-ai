import numpy as np
import pandas as pd
import pytest


def synthetic_raw_sample(n_samples=4000, seed=11):
    rng = np.random.default_rng(seed)
    race = rng.choice([1, 2, 6, 8, 9], size=n_samples, p=[0.55, 0.08, 0.15, 0.15, 0.07])
    income = rng.integers(-5000, 60000, n_samples).astype(float)
    disability = rng.choice([1, 2], n_samples, p=[0.2, 0.8])
    employment = rng.choice([1, 3, 6], n_samples, p=[0.5, 0.2, 0.3])
    score = -income / 15000 + 1.2 * (disability == 1) + 0.6 * (employment != 1) + 0.3 * (race != 1)
    coverage = np.where(score + rng.normal(0, 0.8, n_samples) > 0.2, 1, 2)
    return pd.DataFrame(
        {
            "AGEP": rng.integers(16, 80, n_samples),
            "PINCP": income,
            "SCHL": rng.integers(1, 25, n_samples).astype(float),
            "SEX": rng.choice([1, 2], n_samples),
            "DIS": disability,
            "DEAR": rng.choice([1, 2], n_samples, p=[0.05, 0.95]),
            "DEYE": rng.choice([1, 2], n_samples, p=[0.04, 0.96]),
            "DREM": np.where(rng.random(n_samples) < 0.1, np.nan, rng.choice([1, 2], n_samples, p=[0.08, 0.92])),
            "NATIVITY": rng.choice([1, 2], n_samples, p=[0.75, 0.25]),
            "MAR": rng.choice([1, 2, 3, 4, 5], n_samples),
            "ESP": np.where(rng.random(n_samples) < 0.8, np.nan, rng.choice([1, 2, 3], n_samples)),
            "CIT": rng.choice([1, 4, 5], n_samples, p=[0.7, 0.15, 0.15]),
            "MIG": rng.choice([1, 2, 3], n_samples, p=[0.85, 0.05, 0.1]),
            "MIL": np.where(rng.random(n_samples) < 0.1, np.nan, rng.choice([1, 2, 4], n_samples, p=[0.05, 0.05, 0.9])),
            "ANC": rng.choice([1, 2, 4], n_samples, p=[0.5, 0.3, 0.2]),
            "ESR": employment,
            "ST": np.full(n_samples, 6),
            "FER": np.where(rng.random(n_samples) < 0.5, np.nan, rng.choice([1, 2], n_samples, p=[0.1, 0.9])),
            "RAC1P": race,
            "PUBCOV": coverage,
        }
    )


@pytest.fixture
def acs_raw():
    return synthetic_raw_sample()
