import numpy as np
from sklearn.linear_model import LogisticRegression


def _rates(predictions, labels, group):
    base_rate = float(np.mean(labels))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "accuracy_lift": float(np.mean(predictions == labels)) - max(base_rate, 1.0 - base_rate),
        "positive_rate": float(np.mean(predictions)),
        "positive_rate_reference_group": float(np.mean(predictions[group == 0])),
        "positive_rate_protected_group": float(np.mean(predictions[group == 1])),
    }


def run_diagnostics(model, train, test, blind_indices):
    keep = [index for index in range(train.features.shape[1]) if index not in set(blind_indices)]
    reference = LogisticRegression(max_iter=5000)
    reference.fit(train.features[:, keep], train.labels)
    reference_predictions = reference.predict(test.features[:, keep])
    model_predictions = model.predict(test.features)
    base_rate = float(np.mean(test.labels))
    selected = _rates(model_predictions, test.labels, test.group)
    benchmark = _rates(reference_predictions, test.labels, test.group)
    return {
        "test_size": int(len(test.labels)),
        "base_rate": base_rate,
        "base_rate_reference_group": float(np.mean(test.labels[test.group == 0])),
        "base_rate_protected_group": float(np.mean(test.labels[test.group == 1])),
        "protected_group_share": float(np.mean(test.group)),
        "constant_predictor_accuracy": max(base_rate, 1.0 - base_rate),
        "selected_model": selected,
        "reference_model": benchmark,
        "accuracy_gap_to_reference": benchmark["accuracy"] - selected["accuracy"],
    }
