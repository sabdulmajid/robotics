#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate tracked OpenPI/LIBERO rollout and risk summaries")
    parser.add_argument("--run-dir", default="reports")
    parser.add_argument("--output", default="reports/openpi_results_aggregate.json")
    args = parser.parse_args()

    root = Path(args.run_dir)
    summaries = {}
    for path in sorted(root.glob("openpi_libero_rollout_summary_*.json")):
        summaries[path.stem.removeprefix("openpi_libero_rollout_summary_")] = json.loads(path.read_text())
    risk_path = root / "openpi_libero_risk_summary.json"
    risk = json.loads(risk_path.read_text()) if risk_path.exists() else {}
    payload: dict[str, Any] = {
        "ok": bool(summaries),
        "rollout_summaries": summaries,
        "risk_summary": risk,
        "notes": "Tracked reports only; raw JSONL rollout datasets are generated artifacts under datasets/.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "output": str(output), "rollout_summary_count": len(summaries)}, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
