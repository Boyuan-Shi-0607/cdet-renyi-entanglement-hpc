#!/usr/bin/env python3
"""Plot test-only Green-kernel diagnostics with NumPy and Pillow."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


VIRIDIS_STOPS = np.array(
    [
        [0.00, 68, 1, 84],
        [0.25, 59, 82, 139],
        [0.50, 33, 145, 140],
        [0.75, 94, 201, 98],
        [1.00, 253, 231, 37],
    ],
    dtype=float,
)


def viridis(values: np.ndarray) -> np.ndarray:
    flat = np.clip(values.ravel(), 0.0, 1.0)
    channels = [
        np.interp(flat, VIRIDIS_STOPS[:, 0], VIRIDIS_STOPS[:, channel])
        for channel in range(1, 4)
    ]
    return np.stack(channels, axis=1).reshape((*values.shape, 3)).astype(np.uint8)


def plot_matrix(path: Path) -> None:
    matrix = np.loadtxt(path, delimiter=",")
    positive = matrix[matrix > 0]
    floor = positive.min() if positive.size else np.finfo(float).tiny
    log_matrix = np.log10(np.maximum(matrix, floor))
    low = float(log_matrix.min())
    high = float(log_matrix.max())
    normalized = (log_matrix - low) / (high - low) if high > low else np.zeros_like(log_matrix)
    heatmap = Image.fromarray(viridis(normalized), mode="RGB")
    target_side = min(900, max(360, matrix.shape[0] * 2))
    heatmap = heatmap.resize((target_side, target_side), Image.Resampling.NEAREST)

    margin_left, margin_top, margin_right, margin_bottom = 82, 54, 34, 62
    canvas = Image.new(
        "RGB",
        (margin_left + target_side + margin_right, margin_top + target_side + margin_bottom),
        "white",
    )
    canvas.paste(heatmap, (margin_left, margin_top))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 16), path.stem.replace("middle_abs_", "Scaled middle |M|: "), fill="black")
    draw.text((margin_left + target_side // 2 - 45, margin_top + target_side + 30),
              "eigenmode column", fill="black")
    draw.text((8, margin_top + target_side // 2), "eigenmode row", fill="black")
    draw.text((margin_left, margin_top + target_side + 8), "0", fill="black")
    draw.text((margin_left + target_side - 35, margin_top + target_side + 8),
              str(matrix.shape[1] - 1), fill="black")
    draw.text((4, margin_top), f"log10 range\n{low:.2f}\nto {high:.2f}", fill="black")
    canvas.save(path.with_suffix(".png"))


def plot_conditioning(directory: Path) -> None:
    with (directory / "green_kernel_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    width, height = 900, 560
    left, top, right, bottom = 100, 55, 35, 80
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    plot_width = width - left - right
    plot_height = height - top - bottom
    sizes = sorted({int(row["linear_size"]) for row in rows})
    rconds = [max(float(row["middle_rcond_1"]), 1.0e-300) for row in rows]
    log_low = math.floor(min(math.log10(value) for value in rconds))
    log_high = math.ceil(max(math.log10(value) for value in rconds))
    if log_high == log_low:
        log_high += 1

    def x_coord(size: int) -> float:
        if len(sizes) == 1:
            return left + plot_width / 2
        return left + (size - sizes[0]) / (sizes[-1] - sizes[0]) * plot_width

    def y_coord(value: float) -> float:
        log_value = math.log10(max(value, 1.0e-300))
        return top + (log_high - log_value) / (log_high - log_low) * plot_height

    draw.rectangle((left, top, left + plot_width, top + plot_height), outline="black", width=2)
    for exponent in range(log_low, log_high + 1):
        y = y_coord(10.0**exponent)
        draw.line((left, y, left + plot_width, y), fill=(215, 215, 215), width=1)
        draw.text((12, y - 7), f"1e{exponent}", fill="black")
    for size in sizes:
        x = x_coord(size)
        draw.line((x, top, x, top + plot_height), fill=(235, 235, 235), width=1)
        draw.text((x - 8, top + plot_height + 10), str(size), fill="black")

    styles = [
        ("upper", "0", (33, 113, 181)),
        ("upper", "1", (230, 85, 13)),
        ("lower", "0", (49, 163, 84)),
        ("lower", "1", (117, 107, 177)),
    ]
    legend_y = 18
    for branch, sector, color in styles:
        selected = [
            row for row in rows if row["branch"] == branch and row["sector"] == sector
        ]
        selected.sort(key=lambda row: int(row["linear_size"]))
        points = [
            (x_coord(int(row["linear_size"])), y_coord(float(row["middle_rcond_1"])))
            for row in selected
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        legend_x = left + styles.index((branch, sector, color)) * 180
        draw.line((legend_x, legend_y + 6, legend_x + 24, legend_y + 6), fill=color, width=3)
        draw.text((legend_x + 30, legend_y), f"{branch}, sector {sector}", fill="black")

    draw.text((left + plot_width // 2 - 45, height - 35), "linear lattice size L", fill="black")
    draw.text((8, top - 30), "scaled-middle reciprocal condition number", fill="black")
    canvas.save(directory / "scaled_middle_rcond.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for path in sorted(args.directory.glob("middle_abs_*.csv")):
        plot_matrix(path)
    plot_conditioning(args.directory)


if __name__ == "__main__":
    main()
