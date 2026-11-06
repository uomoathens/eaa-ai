import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


class FairLogisticRegression:
    def __init__(self, n_features, fairness_weight=0.0, explain_weight=0.0, learning_rate=0.1, threshold=0.5, blind_indices=()):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.fairness_weight = float(fairness_weight)
        self.explain_weight = float(explain_weight)
        self.learning_rate = float(learning_rate)
        self.threshold = float(threshold)
        self.mask = np.ones(n_features)
        self.mask[list(blind_indices)] = 0.0

    @property
    def parameters(self):
        return np.concatenate([self.weights, [self.bias]])

    def set_parameters(self, vector):
        vector = np.asarray(vector, dtype=float)
        self.weights = vector[:-1] * self.mask
        self.bias = float(vector[-1])

    def decision_function(self, features):
        return np.asarray(features, dtype=float) @ self.weights + self.bias

    def predict_proba(self, features):
        return _sigmoid(self.decision_function(features))

    def predict(self, features):
        return (self.predict_proba(features) >= self.threshold).astype(int)

    def loss_terms(self, features, labels, group):
        p = self.predict_proba(features)
        eps = 1e-12
        task = float(-np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))
        gap = float(np.mean(p[group == 1]) - np.mean(p[group == 0]))
        fair = gap ** 2
        explain = float(np.abs(self.weights).sum())
        total = task + self.fairness_weight * fair + self.explain_weight * explain
        return {"task": task, "fair": fair, "explain": explain, "total": total, "gap": gap}

    def gradient(self, features, labels, group):
        features = np.asarray(features, dtype=float)
        p = self.predict_proba(features)
        n = len(labels)
        grad_w = features.T @ (p - labels) / n
        grad_b = float(np.mean(p - labels))
        slope = p * (1 - p)
        g1 = group == 1
        g0 = group == 0
        gap = float(np.mean(p[g1]) - np.mean(p[g0]))
        d_gap_w = (features[g1] * slope[g1, None]).mean(axis=0) - (features[g0] * slope[g0, None]).mean(axis=0)
        d_gap_b = float(slope[g1].mean() - slope[g0].mean())
        grad_w = grad_w + self.fairness_weight * 2.0 * gap * d_gap_w + self.explain_weight * np.sign(self.weights)
        grad_b = grad_b + self.fairness_weight * 2.0 * gap * d_gap_b
        return np.concatenate([grad_w * self.mask, [grad_b]])

    def step(self, features, labels, group):
        delta = -self.learning_rate * self.gradient(features, labels, group)
        self.set_parameters(self.parameters + delta)
        return delta
