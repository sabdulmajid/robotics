from __future__ import annotations

import math
from typing import Sequence

from risk_aware_skill_planning.risk.openpi_models import binary_nll


def binary_risk_metrics(labels: Sequence[int], probs: Sequence[float], *, threshold: float = 0.5) -> dict[str, float | None]:
    if not labels:
        return {
            "examples": 0,
            "brier": None,
            "nll": None,
            "ece": None,
            "auroc": None,
            "auprc": None,
            "coverage_at_threshold": None,
            "failure_rate_attempted": None,
        }
    brier = sum((float(label) - prob) ** 2 for label, prob in zip(labels, probs)) / len(labels)
    attempted = [(label, prob) for label, prob in zip(labels, probs) if prob < threshold]
    failures_attempted = sum(label for label, _ in attempted)
    return {
        "examples": len(labels),
        "positive_rate": sum(labels) / len(labels),
        "brier": brier,
        "nll": binary_nll(labels, probs),
        "ece": expected_calibration_error(labels, probs),
        "auroc": auroc(labels, probs),
        "auprc": auprc(labels, probs),
        "coverage_at_threshold": len(attempted) / len(labels),
        "failure_rate_attempted": failures_attempted / len(attempted) if attempted else None,
    }


def expected_calibration_error(labels: Sequence[int], probs: Sequence[float], *, bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        members = [(label, prob) for label, prob in zip(labels, probs) if lo <= prob < hi or (bin_idx == bins - 1 and prob == 1.0)]
        if not members:
            continue
        accuracy = sum(label for label, _ in members) / len(members)
        confidence = sum(prob for _, prob in members) / len(members)
        ece += (len(members) / total) * abs(accuracy - confidence)
    return ece


def reliability_bins(labels: Sequence[int], probs: Sequence[float], *, bins: int = 10) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        members = [(label, prob) for label, prob in zip(labels, probs) if lo <= prob < hi or (bin_idx == bins - 1 and prob == 1.0)]
        output.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "count": len(members),
                "mean_predicted_risk": sum(prob for _, prob in members) / len(members) if members else 0.0,
                "empirical_failure_rate": sum(label for label, _ in members) / len(members) if members else 0.0,
            }
        )
    return output


def auroc(labels: Sequence[int], probs: Sequence[float]) -> float | None:
    positives = [(prob, label) for label, prob in zip(labels, probs) if label == 1]
    negatives = [(prob, label) for label, prob in zip(labels, probs) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos_prob, _ in positives:
        for neg_prob, _ in negatives:
            if pos_prob > neg_prob:
                wins += 1.0
            elif math.isclose(pos_prob, neg_prob):
                wins += 0.5
    return wins / total


def auprc(labels: Sequence[int], probs: Sequence[float]) -> float | None:
    if sum(labels) == 0:
        return None
    pairs = sorted(zip(probs, labels), reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    positives = sum(labels)
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area
