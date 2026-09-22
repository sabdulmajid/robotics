#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the retrospective cross-suite threshold result")
    parser.add_argument("--input", default="reports/openpi_cross_suite_retrospective_threshold_summary.json")
    parser.add_argument("--output", default="reports/figures/openpi_cross_suite_retrospective.svg")
    args = parser.parse_args()

    summary = json.loads(Path(args.input).read_text(encoding="utf-8"))
    write_figure(summary, Path(args.output))
    print(json.dumps({"ok": True, "output": args.output}, indent=2))
    return 0


def write_figure(summary: Mapping[str, Any], output: Path) -> None:
    test = summary["held_out_test"]
    selected = test["selected_threshold"]
    old = test["spatial_threshold_references"]["threshold_0_9860"]["metrics"]
    selected_threshold = float(summary["selection"]["threshold"])
    overall_auroc = float(summary["risk_metrics"]["test"]["auroc"])
    occlusion_auroc = float(summary["risk_metrics"]["test_by_stressor"]["occlusion:0.80"]["auroc"])
    methods = [
        ("Direct OpenPI", test["direct"], "#3f4a5a"),
        (f"Selected {selected_threshold:.4f}", selected, "#167d73"),
        ("Spatial 0.9860", old, "#bf6b21"),
        ("Random abstention", test["random_abstain_matched_coverage"], "#8a8f98"),
    ]
    width, height = 960, 500
    left, top, plot_w, plot_h = 85, 96, 470, 314
    utility_left, utility_w = 650, 230
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 34, "Retrospective cross-suite threshold transfer", 19, "middle", weight="600"),
        text(width / 2, 58, "LIBERO Object and Goal, held-out tasks 7-9, seed 5000", 12, "middle", fill="#4b5563"),
        text(
            width / 2,
            78,
            f"Aggregate AUROC {overall_auroc:.3f}; severe-occlusion AUROC {occlusion_auroc:.3f}",
            12,
            "middle",
            fill="#4b5563",
        ),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(6):
        coverage = 0.5 + tick * 0.1
        x = left + (coverage - 0.5) / 0.5 * plot_w
        failure = tick * 0.06
        y = top + plot_h - failure / 0.30 * plot_h
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eceff1"/>',
                text(x, top + plot_h + 22, f"{coverage:.1f}", 11, "middle"),
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eceff1"/>',
                text(left - 10, y + 4, f"{failure:.2f}", 11, "end"),
            ]
        )
    for index, (name, metrics, color) in enumerate(methods):
        x = left + (float(metrics["coverage"]) - 0.5) / 0.5 * plot_w
        y = top + plot_h - float(metrics["failure_rate_attempted"]) / 0.30 * plot_h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>')
        label_y = top + index * 22
        parts.append(f'<circle cx="{left + plot_w + 22}" cy="{label_y:.1f}" r="6" fill="{color}"/>')
        parts.append(text(left + plot_w + 35, label_y + 4, name, 11))
    parts.extend(
        [
            text(left + plot_w / 2, height - 40, "Coverage", 13, "middle"),
            f'<text x="24" y="{top + plot_h / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2:.1f})">Attempted failure rate</text>',
            text(utility_left + utility_w / 2, top + 120, "Expected utility", 13, "middle", weight="600"),
        ]
    )
    for index, (name, metrics, color) in enumerate(methods[:3]):
        y = top + 150 + index * 58
        value = float(metrics["expected_utility"])
        bar_w = value / 0.70 * utility_w
        parts.append(f'<rect x="{utility_left}" y="{y}" width="{bar_w:.1f}" height="22" fill="{color}"/>')
        parts.append(text(utility_left, y - 7, name, 11))
        parts.append(text(utility_left + bar_w + 7, y + 16, f"{value:.3f}", 11))
    parts.append(text(utility_left + utility_w / 2, top + 350, "Higher utility is better.", 11, "middle", fill="#4b5563"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def text(
    x: float,
    y: float,
    value: str,
    size: int,
    anchor: str = "start",
    *,
    fill: str = "#111827",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="Arial, sans-serif" font-weight="{weight}">{html.escape(value)}</text>'
    )


if __name__ == "__main__":
    raise SystemExit(main())
