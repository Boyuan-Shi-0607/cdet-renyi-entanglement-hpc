#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def read_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def plot_mesh(matrix: np.ndarray,
              x_edges: np.ndarray,
              y_edges: np.ndarray,
              xlabel: str,
              ylabel: str,
              title: str,
              output_path: Path) -> None:
    if plt is None:
        plot_mesh_pil(matrix, x_edges, y_edges, xlabel, ylabel, title, output_path)
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, matrix, shading="flat", cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="conditional probability")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def viridis_like(values: np.ndarray) -> np.ndarray:
    stops = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ], dtype=float)
    values = np.clip(values, 0.0, 1.0)
    scaled = values * (len(stops) - 1)
    low = np.floor(scaled).astype(int)
    high = np.clip(low + 1, 0, len(stops) - 1)
    frac = scaled - low
    rgb = stops[low] * (1.0 - frac[..., None]) + stops[high] * frac[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def plot_mesh_pil(matrix: np.ndarray,
                  x_edges: np.ndarray,
                  y_edges: np.ndarray,
                  xlabel: str,
                  ylabel: str,
                  title: str,
                  output_path: Path) -> None:
    matrix = np.asarray(matrix, dtype=float)
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        normalized = np.zeros_like(matrix, dtype=float)
    else:
        normalized = (matrix - vmin) / (vmax - vmin)

    cell = max(8, min(28, int(520 / max(matrix.shape))))
    heat_h = matrix.shape[0] * cell
    heat_w = matrix.shape[1] * cell
    left = 96
    top = 58
    right = 118
    bottom = 74
    width = left + heat_w + right
    height = top + heat_h + bottom

    image = Image.new("RGB", (width, height), "white")
    heat = Image.fromarray(viridis_like(normalized))
    heat = heat.resize((heat_w, heat_h), Image.Resampling.NEAREST)
    image.paste(heat, (left, top))

    draw = ImageDraw.Draw(image)
    font = load_font(14)
    small = load_font(12)
    title_font = load_font(16)

    draw.rectangle([left, top, left + heat_w, top + heat_h], outline="black", width=1)
    draw.text((left, 18), title, fill="black", font=title_font)
    draw.text((left + heat_w // 2 - 40, top + heat_h + 38), xlabel, fill="black", font=font)
    draw.text((12, top + heat_h // 2), ylabel, fill="black", font=font)

    x_ticks = np.linspace(float(x_edges[0]), float(x_edges[-1]), 5)
    for value in x_ticks:
        x = left + int((value - x_edges[0]) / (x_edges[-1] - x_edges[0]) * heat_w)
        draw.line([(x, top + heat_h), (x, top + heat_h + 5)], fill="black")
        draw.text((x - 14, top + heat_h + 8), f"{value:g}", fill="black", font=small)

    y_ticks = np.linspace(float(y_edges[0]), float(y_edges[-1]), 5)
    for value in y_ticks:
        y = top + heat_h - int((value - y_edges[0]) / (y_edges[-1] - y_edges[0]) * heat_h)
        draw.line([(left - 5, y), (left, y)], fill="black")
        draw.text((left - 48, y - 7), f"{value:g}", fill="black", font=small)

    colorbar_h = heat_h
    colorbar_w = 18
    colorbar_x = left + heat_w + 28
    gradient = np.linspace(1.0, 0.0, colorbar_h)[:, None]
    colorbar = Image.fromarray(viridis_like(gradient))
    colorbar = colorbar.resize((colorbar_w, colorbar_h), Image.Resampling.BILINEAR)
    image.paste(colorbar, (colorbar_x, top))
    draw.rectangle([colorbar_x, top, colorbar_x + colorbar_w, top + colorbar_h],
                   outline="black", width=1)
    draw.text((colorbar_x + 24, top - 4), f"{vmax:.3g}", fill="black", font=small)
    draw.text((colorbar_x + 24, top + colorbar_h - 12), f"{vmin:.3g}", fill="black", font=small)

    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot factorized MCMCMC seed histogram matrices.")
    parser.add_argument("--prefix", required=True,
                        help="Histogram prefix, for example histograms/current_seed")
    args = parser.parse_args()

    prefix = Path(args.prefix)
    metadata = read_metadata(Path(str(prefix) + "_metadata.txt"))
    beta = float(metadata["BETA"])
    n1 = int(metadata["N1"])
    n2 = int(metadata["N2"])
    tau_bins = int(metadata["TAU_BINS"])
    site_count = n1 * n2

    tau = np.loadtxt(Path(str(prefix) + "_tau.csv"), delimiter=",")
    space = np.loadtxt(Path(str(prefix) + "_space.csv"), delimiter=",")
    replica = np.loadtxt(Path(str(prefix) + "_replica.csv"), delimiter=",")

    plot_mesh(tau,
              np.linspace(0.0, beta, tau_bins + 1),
              np.linspace(0.0, beta, tau_bins + 1),
              "child tau'",
              "parent tau",
              "q_tau(child | parent)",
              Path(str(prefix) + "_tau.png"))

    plot_mesh(space,
              np.linspace(0.0, float(site_count), site_count + 1),
              np.linspace(0.0, float(site_count), site_count + 1),
              "child flattened site",
              "parent flattened site",
              "q_space(child | parent)",
              Path(str(prefix) + "_space.png"))

    plot_mesh(replica,
              np.linspace(0.0, 2.0, 3),
              np.linspace(0.0, 2.0, 3),
              "child replica",
              "parent replica",
              "q_rep(child | parent)",
              Path(str(prefix) + "_replica.png"))

    print("Wrote:")
    print(f"  {prefix}_tau.png")
    print(f"  {prefix}_space.png")
    print(f"  {prefix}_replica.png")


if __name__ == "__main__":
    main()
