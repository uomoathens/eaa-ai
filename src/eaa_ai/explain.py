from dataclasses import dataclass

import numpy as np


@dataclass
class Explanation:
    base_value: float
    phi: np.ndarray
    feature_names: list

    def reconstruct(self):
        return self.base_value + self.phi.sum(axis=1)

    def global_importance(self):
        magnitude = np.abs(self.phi).mean(axis=0)
        order = np.argsort(magnitude)[::-1]
        return [(self.feature_names[i], float(magnitude[i])) for i in order]

    def top_k(self, row, k=3):
        values = self.phi[row]
        order = np.argsort(np.abs(values))[::-1][:k]
        return [(self.feature_names[i], float(values[i])) for i in order]

    def share_of(self, indices):
        magnitude = np.abs(self.phi).mean(axis=0)
        total = magnitude.sum()
        return float(magnitude[list(indices)].sum() / total) if total else 0.0


class LinearShapExplainer:
    def __init__(self, background):
        self.background_mean = np.asarray(background, dtype=float).mean(axis=0)

    def explain(self, model, features, feature_names):
        features = np.asarray(features, dtype=float)
        weights = model.weights
        base_value = float(self.background_mean @ weights + model.bias)
        phi = (features - self.background_mean) * weights
        return Explanation(base_value=base_value, phi=phi, feature_names=list(feature_names))
