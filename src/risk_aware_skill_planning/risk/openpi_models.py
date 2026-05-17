from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from risk_aware_skill_planning.risk.openpi_dataset import OpenPIRiskExample


@dataclass(frozen=True)
class OpenPILogisticRiskModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    temperature: float = 1.0

    def predict_proba(self, features: Sequence[float]) -> float:
        z = self.logit(features) / max(self.temperature, 1e-6)
        return _sigmoid(z)

    def logit(self, features: Sequence[float]) -> float:
        normalized = [(float(x) - mu) / sigma for x, mu, sigma in zip(features, self.mean, self.std)]
        return sum(weight * value for weight, value in zip(self.weights, normalized))

    def with_temperature(self, temperature: float) -> "OpenPILogisticRiskModel":
        return OpenPILogisticRiskModel(
            feature_names=self.feature_names,
            weights=self.weights,
            mean=self.mean,
            std=self.std,
            temperature=temperature,
        )


def train_logistic_risk_model(
    examples: Sequence[OpenPIRiskExample],
    *,
    epochs: int = 600,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> OpenPILogisticRiskModel:
    if not examples:
        raise ValueError("At least one training example is required")
    feature_names = examples[0].feature_names
    x_rows = [example.features for example in examples]
    labels = [float(example.label_failure) for example in examples]
    mean, std = _normalization(x_rows)
    x_norm = [[(x - mu) / sigma for x, mu, sigma in zip(row, mean, std)] for row in x_rows]
    weights = [0.0 for _ in feature_names]
    for _ in range(epochs):
        gradients = [0.0 for _ in weights]
        for features, label in zip(x_norm, labels):
            pred = _sigmoid(sum(w * x for w, x in zip(weights, features)))
            error = pred - label
            for idx, value in enumerate(features):
                gradients[idx] += error * value
        scale = 1.0 / len(x_norm)
        for idx, gradient in enumerate(gradients):
            weights[idx] -= learning_rate * (gradient * scale + l2 * weights[idx])
    return OpenPILogisticRiskModel(
        feature_names=feature_names,
        weights=tuple(weights),
        mean=tuple(mean),
        std=tuple(std),
    )


def calibrate_temperature(
    model: OpenPILogisticRiskModel,
    examples: Sequence[OpenPIRiskExample],
    *,
    candidates: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
) -> OpenPILogisticRiskModel:
    if not examples:
        return model
    best_temperature = model.temperature
    best_nll = math.inf
    labels = [example.label_failure for example in examples]
    logits = [model.logit(example.features) for example in examples]
    for temperature in candidates:
        probs = [_sigmoid(logit / temperature) for logit in logits]
        nll = binary_nll(labels, probs)
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return model.with_temperature(best_temperature)


def choose_threshold(
    labels: Sequence[int],
    probs: Sequence[float],
    *,
    failure_penalty: float = 2.0,
    rejection_penalty: float = 0.2,
) -> float:
    if not labels:
        return 0.5
    candidates = sorted({0.0, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, *probs})
    best_threshold = 0.5
    best_utility = -math.inf
    for threshold in candidates:
        utility = 0.0
        for label, prob in zip(labels, probs):
            reject = prob >= threshold
            if reject:
                utility -= rejection_penalty
            elif label:
                utility -= failure_penalty
            else:
                utility += 1.0
        if utility > best_utility:
            best_utility = utility
            best_threshold = threshold
    return float(best_threshold)


def predict_examples(model: OpenPILogisticRiskModel, examples: Sequence[OpenPIRiskExample]) -> list[float]:
    return [model.predict_proba(example.features) for example in examples]


def binary_nll(labels: Sequence[int | float], probs: Sequence[float]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label, prob in zip(labels, probs):
        p = min(max(float(prob), 1e-9), 1.0 - 1e-9)
        total += -(float(label) * math.log(p) + (1.0 - float(label)) * math.log(1.0 - p))
    return total / len(labels)


def _normalization(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    width = len(rows[0])
    means: list[float] = []
    stds: list[float] = []
    for col in range(width):
        values = [row[col] for row in rows]
        mu = sum(values) / len(values)
        variance = sum((value - mu) ** 2 for value in values) / len(values)
        means.append(mu)
        stds.append(max(math.sqrt(variance), 1e-6))
    return means, stds


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
