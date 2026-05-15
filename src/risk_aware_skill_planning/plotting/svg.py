from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping, Sequence


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}">{html.escape(text)}</text>'


def write_reliability_svg(
    reliability_bins: Sequence[Mapping[str, float | int]],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    width = 720
    height = 480
    left = 72
    top = 64
    plot = 340
    bottom = top + plot
    scale = plot
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 30, title, size=18, anchor="middle"),
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot}" y2="{bottom}" stroke="#111" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111" stroke-width="1"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot}" y2="{top}" stroke="#999" stroke-dasharray="4 4"/>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot
        y = bottom - value * plot
        elements.append(f'<line x1="{x}" y1="{bottom}" x2="{x}" y2="{bottom + 5}" stroke="#111"/>')
        elements.append(f'<line x1="{left - 5}" y1="{y}" x2="{left}" y2="{y}" stroke="#111"/>')
        elements.append(_svg_text(x, bottom + 22, f"{value:.1f}", size=11, anchor="middle"))
        elements.append(_svg_text(left - 10, y + 4, f"{value:.1f}", size=11, anchor="end"))
    bin_width = plot / len(reliability_bins)
    for item in reliability_bins:
        mean_probability = float(item["mean_probability"])
        empirical_rate = float(item["empirical_failure_rate"])
        count = int(item["n"])
        x = left + mean_probability * scale
        y = bottom - empirical_rate * scale
        radius = 4 + min(12, count / 60)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#2b6cb0" fill-opacity="0.82"/>')
        bar_x = left + int(item["bin"]) * bin_width
        bar_height = empirical_rate * scale
        elements.append(
            f'<rect x="{bar_x + 2:.1f}" y="{bottom - bar_height:.1f}" width="{bin_width - 4:.1f}" height="{bar_height:.1f}" fill="#90cdf4" fill-opacity="0.35"/>'
        )
    elements.extend(
        [
            _svg_text(left + plot / 2, height - 30, "Predicted failure probability", size=13, anchor="middle"),
            f'<text x="18" y="{top + plot / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 18 {top + plot / 2:.1f})">Empirical failure rate</text>',
            _svg_text(left + plot + 42, top + 34, "Dashed line: perfect calibration", size=12),
            _svg_text(left + plot + 42, top + 58, "Circle size: bin count", size=12),
            "</svg>",
        ]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(elements), encoding="utf-8")


def write_planner_comparison_svg(
    planner_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    width = 960
    height = 520
    left = 92
    top = 72
    plot_width = 760
    plot_height = 320
    scenarios = list(planner_metrics)
    modes = list(next(iter(planner_metrics.values())))
    colors = {
        "naive_no_risk": "#718096",
        "per_skill_prior": "#d69e2e",
        "calibrated_logistic_state_risk": "#2b6cb0",
        "oracle_true_risk": "#2f855a",
        "fixed_per_skill_risk": "#d69e2e",
        "oracle_risk": "#2f855a",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 32, title, size=18, anchor="middle"),
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_height - value * plot_height
        elements.append(f'<line x1="{left - 5}" y1="{y}" x2="{left}" y2="{y}" stroke="#111"/>')
        elements.append(_svg_text(left - 12, y + 4, f"{value:.1f}", size=11, anchor="end"))
        elements.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#eee"/>')
    group_width = plot_width / len(scenarios)
    bar_width = min(34, group_width / (len(modes) * 2.3))
    for scenario_index, scenario in enumerate(scenarios):
        group_left = left + scenario_index * group_width
        center = group_left + group_width / 2
        elements.append(_svg_text(center, top + plot_height + 42, scenario.replace("_", " "), size=11, anchor="middle"))
        for mode_index, mode in enumerate(modes):
            values = planner_metrics[scenario][mode]
            rate = float(values["task_completion_rate"])
            x = center - (len(modes) / 2) * bar_width + mode_index * bar_width * 1.25
            y = top + plot_height - rate * plot_height
            h = rate * plot_height
            color = colors.get(mode, "#4a5568")
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}"/>')
            elements.append(_svg_text(x + bar_width / 2, y - 5, f"{rate:.2f}", size=10, anchor="middle"))
    legend_x = left + plot_width - 250
    legend_y = top - 18
    for index, mode in enumerate(modes):
        y = legend_y + index * 20
        color = colors.get(mode, "#4a5568")
        elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{color}"/>')
        elements.append(_svg_text(legend_x + 18, y, mode, size=11))
    elements.extend(
        [
            f'<text x="22" y="{top + plot_height / 2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">Task completion rate</text>',
            "</svg>",
        ]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(elements), encoding="utf-8")

