#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PANELS = {
    "reward": [
        "reward.overall",
        "reward.answer",
        "reward.format",
        "reward.kinematic",
    ],
    "loss": [
        "actor.pg_loss",
        "actor.kl_loss",
        "actor.entropy_loss",
    ],
    "actor": [
        "actor.grad_norm",
        "actor.lr",
        "actor.ppo_kl",
    ],
    "length": [
        "response_length.mean",
        "prompt_length.mean",
    ],
    "perf": [
        "perf.time_per_step",
        "perf.throughput",
        "perf.max_memory_allocated_gb",
    ],
}

COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (140, 86, 75),
]


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flatten(next_prefix, item, out)
    else:
        out[prefix] = value


def read_rows(log_path):
    rows = []
    if not log_path.exists() or log_path.stat().st_size == 0:
        return rows
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            flat = {}
            flatten("", obj, flat)
            rows.append(flat)
    return normalize_rows(rows)


def normalize_rows(rows):
    keyed = {}
    unkeyed = []
    for order, row in enumerate(rows):
        step = as_float(row.get("step"))
        if step is None:
            unkeyed.append((order, row))
            continue
        keyed[step] = (order, row)

    deduped = [row for _, row in unkeyed]
    deduped.extend(row for _, row in sorted(keyed.values(), key=lambda item: as_float(item[1].get("step"))))
    return deduped


def as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def write_csv(rows, out_path):
    keys = sorted({key for row in rows for key in row})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_panel(rows, metrics, title, out_path):
    points = {}
    steps = [as_float(row.get("step")) for row in rows]
    for metric in metrics:
        series = []
        for row, step in zip(rows, steps):
            value = as_float(row.get(metric))
            if step is not None and value is not None:
                series.append((step, value))
        if series:
            points[metric] = series

    width, height = 1200, 760
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 80, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin_left, 28), title, fill=(20, 20, 20), font=font)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(180, 180, 180),
    )

    if not points:
        draw.text((margin_left + 20, margin_top + 30), "No numeric data yet.", fill=(80, 80, 80), font=font)
        image.save(out_path)
        return

    all_x = [x for series in points.values() for x, _ in series]
    all_y = [y for series in points.values() for _, y in series]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        y_min -= delta
        y_max += delta
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    for i in range(6):
        t = i / 5
        x = margin_left + t * plot_w
        y = margin_top + t * plot_h
        draw.line([(x, margin_top), (x, margin_top + plot_h)], fill=(235, 235, 235))
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill=(235, 235, 235))

        x_val = x_min + t * (x_max - x_min)
        y_val = y_max - t * (y_max - y_min)
        draw.text((x - 15, margin_top + plot_h + 12), f"{x_val:.0f}", fill=(70, 70, 70), font=font)
        draw.text((8, y - 6), f"{y_val:.3g}", fill=(70, 70, 70), font=font)

    draw.text((margin_left + plot_w / 2 - 20, height - 40), "step", fill=(40, 40, 40), font=font)

    def map_x(x):
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def map_y(y):
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    legend_y = margin_top + plot_h + 38
    for idx, (metric, series) in enumerate(points.items()):
        color = COLORS[idx % len(COLORS)]
        mapped = [(map_x(x), map_y(y)) for x, y in series]
        if len(mapped) == 1:
            x, y = mapped[0]
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
        else:
            draw.line(mapped, fill=color, width=3)
            for x, y in mapped:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

        legend_x = margin_left + (idx % 3) * 340
        legend_row_y = legend_y + (idx // 3) * 22
        draw.rectangle([legend_x, legend_row_y + 4, legend_x + 18, legend_row_y + 14], fill=color)
        draw.text((legend_x + 24, legend_row_y), metric, fill=(30, 30, 30), font=font)

    image.save(out_path)


def plot_run(run_dir):
    run_dir = Path(run_dir)
    rows = read_rows(run_dir / "experiment_log.jsonl")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if rows:
        write_csv(rows, plots_dir / "metrics.csv")

    for panel, metrics in PANELS.items():
        draw_panel(rows, metrics, f"{run_dir.name} - {panel}", plots_dir / f"{panel}.png")

    summary = {
        "run_dir": str(run_dir),
        "num_steps_logged": len(rows),
        "first_step": rows[0].get("step") if rows else None,
        "last_step": rows[-1].get("step") if rows else None,
        "plots_dir": str(plots_dir),
    }
    (plots_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="EasyR1 checkpoint directories containing experiment_log.jsonl")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        summary = plot_run(run_dir)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
