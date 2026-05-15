from __future__ import annotations

from typing import Mapping


def metrics_to_markdown_table(metrics: Mapping[str, float | int]) -> str:
    rows = ["| metric | value |", "| --- | ---: |"]
    for key, value in metrics.items():
        if isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        rows.append(f"| {key} | {rendered} |")
    return "\n".join(rows)

