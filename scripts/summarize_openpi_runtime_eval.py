#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from risk_aware_skill_planning.evaluation.bootstrap import bootstrap_ci
from risk_aware_skill_planning.evaluation.risk_coverage import expected_utility


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize runtime OpenPI/LIBERO supervisor JSONL files")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL path or glob; may be repeated")
    parser.add_argument("--output", default="reports/openpi_runtime_siglip_eval_summary.json")
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    episodes = [episode for path in paths for episode in read_jsonl(path)]
    summary = summarize_runtime_episodes(episodes)
    summary["input_paths"] = [str(path) for path in paths]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "episodes": summary["episodes"], "output": str(output)}, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


def summarize_runtime_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        by_mode.setdefault(str(episode.get("mode", "unknown")), []).append(episode)
    mode_summaries = {mode: summarize_mode(items) for mode, items in sorted(by_mode.items())}
    direct_queries = mode_summaries.get("direct_openpi", {}).get("mean_policy_queries")
    if direct_queries:
        for payload in mode_summaries.values():
            payload["policy_query_overhead_vs_direct"] = payload.get("mean_policy_queries", 0.0) / max(float(direct_queries), 1e-9)
    return {
        "ok": bool(episodes),
        "episodes": len(episodes),
        "modes": mode_summaries,
        "by_suite": count_by(episodes, "libero_suite"),
        "by_stressor": count_by(episodes, "stressor_name"),
        "by_stressor_severity": count_by_stressor_severity(episodes),
        "gpu_models": count_by(episodes, "cuda_device_name"),
    }


def summarize_mode(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(episodes)
    outcomes = [episode_outcome(episode) for episode in episodes]
    success_values = [float(item["success"]) for item in outcomes]
    failure_values = [float(item["failure"]) for item in outcomes]
    timeout_values = [float(item["timeout"]) for item in outcomes]
    abstain_values = [float(item["abstained"]) for item in outcomes]
    utility_values = [float(item["utility"]) for item in outcomes]
    query_values = [float(item["policy_queries"]) for item in outcomes]
    risk_time_values = [float(item["risk_compute_seconds"]) for item in outcomes if item["risk_compute_seconds"] is not None]
    attempted = max(1.0, total - sum(abstain_values))
    return {
        "episodes": total,
        "coverage": 1.0 - sum(abstain_values) / total if total else 0.0,
        "task_completion_rate": sum(success_values) / total if total else 0.0,
        "failure_rate": sum(failure_values) / total if total else 0.0,
        "failure_rate_attempted": sum(failure_values) / attempted,
        "timeout_rate": sum(timeout_values) / total if total else 0.0,
        "abstention_rate": sum(abstain_values) / total if total else 0.0,
        "expected_utility": sum(utility_values) / total if total else 0.0,
        "mean_policy_queries": sum(query_values) / total if total else 0.0,
        "mean_runtime_risk_compute_seconds": mean(risk_time_values) if risk_time_values else 0.0,
        "ci95": {
            "task_completion_rate": bootstrap_ci(success_values, mean_float, samples=500),
            "failure_rate": bootstrap_ci(failure_values, mean_float, samples=500),
            "timeout_rate": bootstrap_ci(timeout_values, mean_float, samples=500),
            "abstention_rate": bootstrap_ci(abstain_values, mean_float, samples=500),
            "expected_utility": bootstrap_ci(utility_values, mean_float, samples=500),
        },
    }


def episode_outcome(episode: Mapping[str, Any]) -> dict[str, Any]:
    terminal = str(episode.get("terminal_label", episode.get("terminal_reason", "")))
    failure_label = str(episode.get("failure_label", ""))
    abstained = terminal == "abstained" or failure_label == "abstained"
    success = bool(episode.get("success", False)) and not abstained
    timeout = bool(episode.get("timeout", False)) or terminal == "timeout"
    failure = (not success) and (not abstained)
    episode_length = int(episode.get("episode_length", 0) or 0)
    metadata = episode.get("metadata", {}) if isinstance(episode.get("metadata"), Mapping) else {}
    decision = episode.get("runtime_supervisor_decision") or metadata.get("runtime_supervisor_decision")
    risk_compute_seconds = decision.get("risk_compute_seconds") if isinstance(decision, Mapping) else None
    return {
        "success": success,
        "failure": failure,
        "timeout": timeout,
        "abstained": abstained,
        "utility": expected_utility(success=success, timeout=timeout, abstained=abstained, episode_length=episode_length),
        "policy_queries": int(metadata.get("action_queries", 0) or 0),
        "risk_compute_seconds": risk_compute_seconds,
    }


def expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    return sorted(dict.fromkeys(paths))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_by(episodes: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for episode in episodes:
        value = str(episode.get(key, "unknown"))
        output[value] = output.get(value, 0) + 1
    return dict(sorted(output.items()))


def count_by_stressor_severity(episodes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for episode in episodes:
        params = episode.get("stressor_params", {})
        severity = float(params.get("severity", 0.0)) if isinstance(params, Mapping) else 0.0
        key = f"{episode.get('stressor_name', 'unknown')}:{severity:.2f}"
        output[key] = output.get(key, 0) + 1
    return dict(sorted(output.items()))


def mean_float(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
