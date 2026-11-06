import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    group: np.ndarray
    feature_names: list
    sensitive_index: int

    def subset(self, index):
        return Dataset(
            features=self.features[index],
            labels=self.labels[index],
            group=self.group[index],
            feature_names=self.feature_names,
            sensitive_index=self.sensitive_index,
        )

    def fingerprint(self):
        digest = hashlib.sha256()
        for array in (self.features, self.labels, self.group):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()


def synthetic_public_benefit(n_samples=6000, historical_bias=0.8, proxy_strength=0.7, seed=7):
    rng = np.random.default_rng(seed)
    group = rng.binomial(1, 0.35, n_samples)
    rural = np.where(rng.random(n_samples) < proxy_strength, group, rng.binomial(1, 0.35, n_samples))
    income = rng.normal(0.0, 1.0, n_samples)
    household = rng.poisson(2.5, n_samples).astype(float)
    employment = np.clip(rng.normal(8.0, 5.0, n_samples), 0.0, None)
    digital = rng.binomial(1, np.where(rural == 1, 0.55, 0.85))
    need = -1.1 * income + 0.35 * household - 0.05 * employment + 0.3 * rng.normal(size=n_samples)
    labels = (need - historical_bias * group > np.quantile(need, 0.55)).astype(int)
    frame = pd.DataFrame(
        {
            "income": income,
            "household_size": household,
            "employment_years": employment,
            "digital_access": digital,
            "rural_region": rural,
            "protected_group": group,
            "eligible": labels,
        }
    )
    return frame


def load_table(path):
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    return pd.read_csv(source)


def frame_to_dataset(frame, target, sensitive, features=None):
    columns = list(features) if features else [column for column in frame.columns if column != target]
    if sensitive not in columns:
        columns.append(sensitive)
    matrix = frame[columns].to_numpy(dtype=float)
    return Dataset(
        features=matrix,
        labels=frame[target].to_numpy(dtype=int),
        group=frame[sensitive].to_numpy(dtype=int),
        feature_names=columns,
        sensitive_index=columns.index(sensitive),
    )


def split_and_standardise(dataset, fractions=(0.6, 0.2, 0.2), seed=7):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.labels))
    first = int(fractions[0] * len(order))
    second = first + int(fractions[1] * len(order))
    train, validation, test = (dataset.subset(order[:first]), dataset.subset(order[first:second]), dataset.subset(order[second:]))
    binary = [
        index
        for index in range(train.features.shape[1])
        if set(np.unique(train.features[:, index])).issubset({0.0, 1.0})
    ]
    mean = train.features.mean(axis=0)
    scale = train.features.std(axis=0)
    mean[binary] = 0.0
    scale[binary] = 1.0
    scale[scale == 0.0] = 1.0
    for part in (train, validation, test):
        part.features = (part.features - mean) / scale
    return train, validation, test
