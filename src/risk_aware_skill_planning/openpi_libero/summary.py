from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from risk_aware_skill_planning.openpi_libero.schema import read_episode_jsonl, summarize_episode_logs


def summarize_rollout_file(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    summary = summarize_episode_logs(read_episode_jsonl(input_path))
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
