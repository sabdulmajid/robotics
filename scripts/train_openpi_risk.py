#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from risk_aware_skill_planning.evaluation.openpi_risk import run_openpi_risk_training, write_openpi_risk_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and calibrate an OpenPI/LIBERO risk critic")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config.get("data", {})
    outputs = config.get("outputs", {})
    summary = run_openpi_risk_training(
        [str(pattern) for pattern in data.get("inputs", [])],
        prefix_steps=int(data.get("prefix_steps", 10)),
    )
    write_openpi_risk_outputs(
        summary,
        outputs.get("summary_path", "reports/openpi_libero_risk_summary.json"),
        outputs.get("report_path", "reports/openpi_libero_risk_planning.md"),
    )
    print(json.dumps({"ok": summary.get("ok", False), "blocker": summary.get("blocker")}, indent=2, sort_keys=True))
    return 0 if summary.get("ok", False) else 2


def load_config(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return config


if __name__ == "__main__":
    raise SystemExit(main())
