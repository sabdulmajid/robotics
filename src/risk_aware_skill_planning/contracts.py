from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


TERMINAL_SUCCESS = "success"
TERMINAL_TASK_FAILURE = "task_failure"
TERMINAL_SAFETY_FAILURE = "safety_failure"
TERMINAL_TIMEOUT = "timeout"
TERMINAL_REJECTED = "unsafe_or_uncertain"


@dataclass(frozen=True)
class FeatureSpec:
    version: str
    field_names: tuple[str, ...]
    field_shapes: Mapping[str, tuple[int, ...]]
    normalization_mean: Mapping[str, float]
    normalization_std: Mapping[str, float]
    angle_representation: str
    object_id_encoding: Mapping[str, int]
    target_id_encoding: Mapping[str, int]
    variant_id_encoding: Mapping[str, int]

    def validate(self) -> None:
        missing_shapes = set(self.field_names) - set(self.field_shapes)
        missing_mean = set(self.field_names) - set(self.normalization_mean)
        missing_std = set(self.field_names) - set(self.normalization_std)
        if missing_shapes or missing_mean or missing_std:
            raise ValueError(
                "FeatureSpec is missing entries for "
                f"shapes={sorted(missing_shapes)}, "
                f"mean={sorted(missing_mean)}, std={sorted(missing_std)}"
            )
        bad_std = [name for name, value in self.normalization_std.items() if value <= 0.0]
        if bad_std:
            raise ValueError(f"normalization_std must be positive for {bad_std}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointMetadata:
    feature_spec_version: str
    normalization_stats: Mapping[str, Mapping[str, float]]
    dataset_hash: str
    policy_checkpoint_id: str
    git_commit_or_run_id: str
    config: Mapping[str, Any]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventFlags:
    miss: bool = False
    drop: bool = False
    high_force_collision: bool = False
    wrong_location: bool = False
    no_progress: bool = False
    monitor_triggered: bool = False

    def any(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutOutcome:
    terminal_label: str
    event_flags: EventFlags = field(default_factory=EventFlags)
    first_failure_event: str | None = None
    failure_time: int | None = None

    @classmethod
    def success(cls) -> "RolloutOutcome":
        return cls(terminal_label=TERMINAL_SUCCESS)

    @property
    def any_failure(self) -> bool:
        return self.terminal_label != TERMINAL_SUCCESS

    @property
    def task_failure(self) -> bool:
        return self.terminal_label == TERMINAL_TASK_FAILURE

    @property
    def safety_failure(self) -> bool:
        return self.terminal_label == TERMINAL_SAFETY_FAILURE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_flags"] = self.event_flags.to_dict()
        return data


@dataclass(frozen=True)
class SkillCall:
    skill_id: str
    object_id: str = ""
    subgoal_id: str = ""
    variant_id: str = ""

    @property
    def action_id(self) -> str:
        if self.skill_id == "pick":
            return f"{self.variant_id}_pick"
        if self.skill_id == "place":
            return f"{self.variant_id}_place"
        if self.skill_id == "move_distractor":
            return "move_distractor"
        if self.skill_id == "recover_safe_pose":
            return "recover"
        return self.skill_id

    def display(self) -> str:
        parts = [self.skill_id]
        if self.object_id:
            parts.append(self.object_id)
        if self.subgoal_id:
            parts.append(self.subgoal_id)
        if self.variant_id:
            parts.append(self.variant_id)
        return "(" + ", ".join(parts) + ")"

    def to_dict(self) -> dict[str, str]:
        return asdict(self) | {"action_id": self.action_id}


@dataclass(frozen=True)
class RiskEstimate:
    p_any_failure: float
    p_task_failure: float | None = None
    p_safety_failure: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("p_any_failure", self.p_any_failure),
            ("p_task_failure", self.p_task_failure),
            ("p_safety_failure", self.p_safety_failure),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def task(self) -> float:
        return self.p_task_failure if self.p_task_failure is not None else self.p_any_failure

    @property
    def safety(self) -> float:
        return self.p_safety_failure if self.p_safety_failure is not None else self.p_any_failure

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


class RiskModel(Protocol):
    def predict(self, state_features: Any, skill: SkillCall) -> RiskEstimate:
        ...


@dataclass(frozen=True)
class SkillSpec:
    skill_call: SkillCall
    cost: float
    preconditions: Callable[[Any], bool]
    postcondition: Callable[[Any], bool]


@dataclass(frozen=True)
class CandidatePlanLog:
    plan: tuple[SkillCall, ...]
    action_cost_sum: float
    next_skill_risk: float
    tail_risk_score: float
    risk_union_bound_score: float
    success_product_score: float
    total_cost: float
    rejected_by_threshold: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": [skill.to_dict() for skill in self.plan],
            "action_cost_sum": self.action_cost_sum,
            "next_skill_risk": self.next_skill_risk,
            "tail_risk_score": self.tail_risk_score,
            "risk_union_bound_score": self.risk_union_bound_score,
            "success_product_score": self.success_product_score,
            "total_cost": self.total_cost,
            "rejected_by_threshold": self.rejected_by_threshold,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class PlanningDecision:
    selected_plan: tuple[SkillCall, ...] | None
    candidate_logs: tuple[CandidatePlanLog, ...]
    rejected: bool
    rejection_reason: str | None = None

    def selected_next_skill(self) -> SkillCall | None:
        if not self.selected_plan:
            return None
        return self.selected_plan[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_plan": None
            if self.selected_plan is None
            else [skill.to_dict() for skill in self.selected_plan],
            "candidate_logs": [log.to_dict() for log in self.candidate_logs],
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class EpisodeResult:
    scenario_id: str
    planner_mode: str
    seed: int
    terminal_label: str
    attempted: bool
    rejected: bool
    rejection_reason: str | None
    execution_cost: float
    cumulative_predicted_risk: float
    replans: int
    failed_skill_executions: int
    executed_skills: list[SkillCall] = field(default_factory=list)
    outcomes: list[RolloutOutcome] = field(default_factory=list)
    planning_logs: list[PlanningDecision] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.terminal_label == TERMINAL_SUCCESS

    @property
    def safety_failure(self) -> bool:
        return self.terminal_label == TERMINAL_SAFETY_FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "planner_mode": self.planner_mode,
            "seed": self.seed,
            "terminal_label": self.terminal_label,
            "attempted": self.attempted,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "execution_cost": self.execution_cost,
            "cumulative_predicted_risk": self.cumulative_predicted_risk,
            "replans": self.replans,
            "failed_skill_executions": self.failed_skill_executions,
            "executed_skills": [skill.to_dict() for skill in self.executed_skills],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "planning_logs": [decision.to_dict() for decision in self.planning_logs],
        }


def risk_union_bound_score(risks: Sequence[float]) -> float:
    return min(1.0, sum(risks))


def success_product_score(risks: Sequence[float]) -> float:
    product = 1.0
    for risk in risks:
        product *= 1.0 - risk
    return product

