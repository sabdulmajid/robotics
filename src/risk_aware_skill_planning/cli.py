from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from risk_aware_skill_planning.configs import load_experiment_config
from risk_aware_skill_planning.envs.toy import ToySymbolicEnv
from risk_aware_skill_planning.evaluation.risk_eval import (
    run_toy_risk_learning,
    write_json_summary,
    write_toy_risk_report,
)
from risk_aware_skill_planning.evaluation.toy_eval import run_toy_suite
from risk_aware_skill_planning.plotting import write_planner_comparison_svg, write_reliability_svg
from risk_aware_skill_planning.planning.toy_planner import PlannerConfig, ToyPlanner, run_toy_episode


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def cmd_smoke(_: argparse.Namespace) -> int:
    env = ToySymbolicEnv("direct_pick_blocked_by_distractor")
    state = env.reset(seed=0)
    planner = ToyPlanner(PlannerConfig(mode="oracle_risk"))
    decision = planner.plan(state)
    print(
        json.dumps(
            {
                "ok": True,
                "scenario_id": env.scenario_id,
                "state": state.to_dict(),
                "selected_plan": None
                if decision.selected_plan is None
                else [skill.to_dict() for skill in decision.selected_plan],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    env = ToySymbolicEnv(config["scenario_ids"][0])
    state = env.reset(seed=int(config.get("seed_start", 0)))
    decisions = {}
    for mode in config["planner_modes"]:
        planner_cfg = PlannerConfig(mode=mode, **config["planner"])
        planner = ToyPlanner(planner_cfg)
        decisions[mode] = planner.plan(state).to_dict()
    print(
        json.dumps(
            {
                "ok": True,
                "experiment_id": config["experiment_id"],
                "scenario_ids": config["scenario_ids"],
                "planner_modes": config["planner_modes"],
                "sample_decisions": decisions,
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


def cmd_toy_eval(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    planner = config["planner"]
    summary = run_toy_suite(
        config["scenario_ids"],
        config["planner_modes"],
        num_episodes=config["num_episodes"],
        seed_start=int(config.get("seed_start", 0)),
        lambda_risk=float(planner["lambda_risk"]),
        next_skill_threshold=float(planner["next_skill_threshold"]),
        plan_threshold=float(planner["plan_threshold"]),
        max_replans=int(planner["max_replans"]),
    )
    output_path = Path(config["outputs"]["summary_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps({"ok": True, "summary_path": str(output_path)}, indent=2, sort_keys=True))
    return 0


def cmd_toy_trace(args: argparse.Namespace) -> int:
    planner = ToyPlanner(
        PlannerConfig(
            mode=args.planner_mode,
            lambda_risk=args.lambda_risk,
            next_skill_threshold=args.next_skill_threshold,
            plan_threshold=args.plan_threshold,
            max_replans=args.max_replans,
        )
    )
    episode = run_toy_episode(
        scenario_id=args.scenario,
        planner=planner,
        seed=args.seed,
        max_replans=args.max_replans,
    )
    payload = episode.to_dict()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "trace_path": str(output_path),
                "terminal_label": episode.terminal_label,
                "executed_skills": [skill.action_id for skill in episode.executed_skills],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_toy_risk_eval(args: argparse.Namespace) -> int:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {args.config} did not parse to a mapping")
    outputs = config["outputs"]
    summary = run_toy_risk_learning(
        seed=int(config["seed"]),
        train_examples=int(config["train_examples"]),
        calibration_examples=int(config["calibration_examples"]),
        test_examples=int(config["test_examples"]),
        planner_eval_episodes=int(config["planner_eval_episodes"]),
        risk_threshold=float(config["risk_threshold"]),
    )
    summary_path = Path(outputs["summary_path"])
    report_path = Path(outputs["report_path"])
    figures_dir = Path(outputs["figures_dir"])
    write_json_summary(summary, summary_path)
    write_toy_risk_report(summary, report_path)
    calibrated_bins = summary["risk_metrics"]["calibrated_logistic_state_risk"]["reliability_bins"]
    write_reliability_svg(
        calibrated_bins,
        figures_dir / "toy_calibrated_risk_reliability.svg",
        title="Toy calibrated state-risk reliability",
    )
    write_planner_comparison_svg(
        summary["planner_metrics"],
        figures_dir / "toy_learned_risk_planner_comparison.svg",
        title="Toy planner task completion",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "figures": [
                    str(figures_dir / "toy_calibrated_risk_reliability.svg"),
                    str(figures_dir / "toy_learned_risk_planner_comparison.svg"),
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Risk-aware skill planning utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Import package, reset toy env, and plan one step")
    smoke.set_defaults(func=cmd_smoke)

    dry_run = subparsers.add_parser("dry-run", help="Validate an experiment config without training")
    dry_run.add_argument("--config", required=True)
    dry_run.set_defaults(func=cmd_dry_run)

    toy_eval = subparsers.add_parser("toy-eval", help="Run the frozen toy scenario suite")
    toy_eval.add_argument("--config", required=True)
    toy_eval.set_defaults(func=cmd_toy_eval)

    toy_trace = subparsers.add_parser("toy-trace", help="Write one full toy episode trace with candidate-plan logs")
    toy_trace.add_argument("--scenario", default="direct_pick_blocked_by_distractor")
    toy_trace.add_argument("--planner-mode", default="oracle_risk")
    toy_trace.add_argument("--seed", type=int, default=0)
    toy_trace.add_argument("--lambda-risk", type=float, default=3.0)
    toy_trace.add_argument("--next-skill-threshold", type=float, default=0.85)
    toy_trace.add_argument("--plan-threshold", type=float, default=0.85)
    toy_trace.add_argument("--max-replans", type=int, default=8)
    toy_trace.add_argument("--output", default="outputs/toy_trace.json")
    toy_trace.set_defaults(func=cmd_toy_trace)

    toy_risk_eval = subparsers.add_parser("toy-risk-eval", help="Train/evaluate toy learned risk baselines")
    toy_risk_eval.add_argument("--config", required=True)
    toy_risk_eval.set_defaults(func=cmd_toy_risk_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
