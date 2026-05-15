from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from risk_aware_skill_planning.contracts import RiskEstimate, SkillCall
from risk_aware_skill_planning.datasets.toy_risk import ToyRiskExample
from risk_aware_skill_planning.envs.toy import ToyState


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class GlobalPriorRiskModel:
    probability: float

    @classmethod
    def fit(cls, examples: Sequence[ToyRiskExample]) -> "GlobalPriorRiskModel":
        failures = sum(example.any_failure for example in examples)
        probability = (failures + 1.0) / (len(examples) + 2.0)
        return cls(probability=probability)

    def predict(self, state_features: ToyState, skill: SkillCall) -> RiskEstimate:
        return RiskEstimate(_clip_probability(self.probability))


@dataclass(frozen=True)
class EmpiricalPerSkillRiskModel:
    probabilities: Mapping[str, float]
    fallback_probability: float

    @classmethod
    def fit(cls, examples: Sequence[ToyRiskExample]) -> "EmpiricalPerSkillRiskModel":
        global_model = GlobalPriorRiskModel.fit(examples)
        counts: dict[str, list[int]] = {}
        for example in examples:
            values = counts.setdefault(example.skill.action_id, [0, 0])
            values[0] += 1
            values[1] += int(example.any_failure)
        probabilities = {
            action_id: (failures + 1.0) / (n + 2.0)
            for action_id, (n, failures) in counts.items()
        }
        return cls(probabilities=probabilities, fallback_probability=global_model.probability)

    def predict(self, state_features: ToyState, skill: SkillCall) -> RiskEstimate:
        risk = self.probabilities.get(skill.action_id, self.fallback_probability)
        return RiskEstimate(_clip_probability(risk))


def toy_risk_feature_names() -> tuple[str, ...]:
    state_names = (
        "object_blocked",
        "object_far",
        "gripper_empty",
        "holding_object",
        "at_safe_pose",
        "distractor_clear",
        "object_at_goal",
        "timestep",
    )
    action_names = (
        "direct_pick",
        "conservative_pick",
        "move_distractor",
        "fast_place",
        "slow_place",
        "recover",
    )
    names = ["bias"]
    names.extend(f"state:{name}" for name in state_names)
    names.extend(f"action:{name}" for name in action_names)
    for action in action_names:
        for state in state_names:
            names.append(f"interaction:{action}:{state}")
    return tuple(names)


def toy_risk_features(state: ToyState, skill: SkillCall) -> tuple[float, ...]:
    state_values = (
        float(state.object_blocked),
        float(state.object_far),
        float(state.gripper_empty),
        float(state.holding_object),
        float(state.at_safe_pose),
        float(state.distractor_clear),
        float(state.object_at_goal),
        float(state.timestep) / 3.0,
    )
    action_names = (
        "direct_pick",
        "conservative_pick",
        "move_distractor",
        "fast_place",
        "slow_place",
        "recover",
    )
    action_values = tuple(float(skill.action_id == action) for action in action_names)
    interactions = tuple(action_value * state_value for action_value in action_values for state_value in state_values)
    return (1.0, *state_values, *action_values, *interactions)


@dataclass(frozen=True)
class LogisticRiskModel:
    weights: tuple[float, ...]
    temperature: float = 1.0

    @classmethod
    def fit(
        cls,
        examples: Sequence[ToyRiskExample],
        *,
        learning_rate: float = 0.18,
        epochs: int = 120,
        l2: float = 0.002,
    ) -> "LogisticRiskModel":
        feature_count = len(toy_risk_feature_names())
        weights = [0.0] * feature_count
        n = float(len(examples))
        if n == 0.0:
            raise ValueError("Cannot fit LogisticRiskModel with no examples")
        for _ in range(epochs):
            gradients = [0.0] * feature_count
            for example in examples:
                features = toy_risk_features(example.state, example.skill)
                probability = _sigmoid(sum(weight * feature for weight, feature in zip(weights, features)))
                error = probability - float(example.any_failure)
                for index, feature in enumerate(features):
                    gradients[index] += error * feature
            for index in range(feature_count):
                penalty = l2 * weights[index] if index != 0 else 0.0
                weights[index] -= learning_rate * (gradients[index] / n + penalty)
        return cls(weights=tuple(weights))

    def logit(self, state: ToyState, skill: SkillCall) -> float:
        features = toy_risk_features(state, skill)
        raw_logit = sum(weight * feature for weight, feature in zip(self.weights, features))
        return raw_logit / self.temperature

    def predict(self, state_features: ToyState, skill: SkillCall) -> RiskEstimate:
        return RiskEstimate(_clip_probability(_sigmoid(self.logit(state_features, skill))))

    def with_temperature(self, temperature: float) -> "LogisticRiskModel":
        return LogisticRiskModel(weights=self.weights, temperature=temperature)


def negative_log_likelihood(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        return 0.0
    total = 0.0
    for probability, label in zip(probabilities, labels):
        p = _clip_probability(probability)
        total -= math.log(p if label else 1.0 - p)
    return total / len(probabilities)


def fit_temperature(
    model: LogisticRiskModel,
    calibration_examples: Sequence[ToyRiskExample],
    *,
    candidates: Sequence[float] | None = None,
) -> LogisticRiskModel:
    if candidates is None:
        candidates = tuple(0.5 + 0.05 * index for index in range(91))
    labels = [example.any_failure for example in calibration_examples]
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in candidates:
        calibrated = model.with_temperature(temperature)
        probabilities = [calibrated.predict(example.state, example.skill).p_any_failure for example in calibration_examples]
        nll = negative_log_likelihood(probabilities, labels)
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return model.with_temperature(best_temperature)
