"""Plot the sampling priors in distributions.yaml, one PDF panel per effect.

distributions.yaml is a nested tree grouped by workflow family rather than by effect, so the
unit plotted here is the level-2 `<family>.<effect>` node (e.g. `vocals.compression`). Every
leaf distribution beneath it becomes one subplot titled "<parameter>: <type>", and the panel is
written to <output-dir>/<family>_<effect>.pdf.

A `joint` leaf is a weighted mixture over whole parameter sets, so it expands into one subplot
per parameter showing that parameter's mixture marginal, resampled onto a common grid (its
components can disagree on both bin edges and kind).

Rendering is fully deterministic: every kind is drawn from its declared weights or its
closed-form density, so no RNG is involved and the PDFs are byte-reproducible across runs.
"""

import math
import argparse
from pathlib import Path
from collections.abc import Mapping, Sequence
import yaml
import matplotlib
from typing import Any

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm.auto import tqdm

CONFIG_DIR = Path("configs/ground_truth")
OUTPUT_DIR = Path("/dartfs/rc/lab/S/SinghN/projects/rime_artifacts/figures/distribution_priors")

# One series per subplot, so a single categorical slot is all that is needed. Landmarks and
# text wear ink tokens rather than a second hue: status colors stay reserved for status.
SERIES = "#2a78d6"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"

# Bins used when a joint's components must be resampled onto one common grid.
MARGINAL_BINS = 24


def stage(message: str) -> None:
    tqdm.write(f"[plot-distributions] {message}")


def format_value(value: Any) -> str:
    """Trim trailing zeros so ticks read 3800 and -3 rather than 3800.0 and -3.0."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    return "%g" % round(value, 4 if abs(value) < 100.0 else 1)


def is_leaf(node: Mapping[str, Any]) -> bool:
    """Leaf test copied from GroundTruthPlanner._looks_like_distribution (ground_truth/planner.py)."""
    return "type" in node or "values" in node or ("low" in node and "high" in node)


def distribution_kind(spec: Mapping[str, Any]) -> str:
    """The planner defaults an untyped spec to choice; mirror that so titles never read 'None'."""
    return str(spec.get("type", "choice" if "values" in spec else "uniform"))


def choice_values(spec: Mapping[str, Any]) -> list[tuple[Any, float]]:
    """(value, weight) pairs; mirrors GroundTruthPlanner._choice_values, bare scalars weigh 1.0."""
    pairs = [(item["value"], float(item.get("weight", 1.0))) if isinstance(item, Mapping) and "value" in item else (item, 1.0) for item in spec.get("values", [])]
    numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value, _ in pairs)
    return sorted(pairs, key=lambda pair: pair[0]) if numeric else pairs


def joint_leaves(spec: Mapping[str, Any], prefix: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Expand a joint into one pseudo-leaf per parameter, carrying that parameter's mixture."""
    names: list[str] = []
    for component in spec.get("components", []):
        for name in component.get("parameters", {}):
            if name not in names:
                names.append(name)
    leaves = []
    for name in names:
        parts = [(float(component.get("weight", 1.0)), component["parameters"][name]) for component in spec["components"] if name in component.get("parameters", {})]
        leaves.append(("%s.%s" % (prefix, name) if prefix else name, {"type": "joint", "parts": parts}))
    return leaves


def collect_leaves(node: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    """(name relative to the effect node, leaf spec) pairs, in file order."""
    if is_leaf(node):
        return joint_leaves(node, prefix) if node.get("type") == "joint" else [(prefix, node)]
    leaves: list[tuple[str, Mapping[str, Any]]] = []
    for key, child in node.items():
        if isinstance(child, Mapping):
            leaves.extend(collect_leaves(child, "%s.%s" % (prefix, key) if prefix else str(key)))
    return leaves


def effect_nodes(distributions: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Level-2 `family.effect` nodes in file order; a level-1 leaf stands in as its own effect."""
    nodes: list[tuple[str, Mapping[str, Any]]] = []
    for family_name, family in distributions.items():
        if not isinstance(family, Mapping):
            continue
        if is_leaf(family):
            nodes.append((str(family_name), family))
            continue
        for effect_name, effect in family.items():
            if isinstance(effect, Mapping):
                nodes.append(("%s.%s" % (family_name, effect_name), effect))
    return nodes


def style_axes(axes: Axes) -> None:
    """Recessive grid and axes so the marks carry the panel."""
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(GRID)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.6)


def set_edge_ticks(axes: Axes, edges: Sequence[float], log_scale: bool) -> None:
    """Label at most five bin edges, so a 24-bin marginal keeps a readable axis."""
    if log_scale and edges[0] > 0.0:
        axes.set_xscale("log")
        axes.minorticks_off()
    step = max(1, (len(edges) - 1) // 4)
    ticks = list(edges[::step])
    if ticks[-1] != edges[-1]:
        ticks.append(edges[-1])
    axes.set_xticks(ticks)
    axes.set_xticklabels([format_value(value) for value in ticks])


def draw_bins(axes: Axes, edges: Sequence[float], mass: Sequence[float], log_scale: bool) -> None:
    """Shared bar drawing for anything already reduced to (bin edges, probability mass)."""
    widths = [edges[index + 1] - edges[index] for index in range(len(mass))]
    axes.bar(edges[:-1], mass, width=widths, align="edge", color=SERIES, edgecolor=SURFACE, linewidth=0.5)
    axes.set_xlim(edges[0], edges[-1])
    axes.set_ylim(0.0, max(mass) * 1.2 if max(mass) > 0.0 else 1.0)
    set_edge_ticks(axes, edges, log_scale)
    axes.set_ylabel("probability", color=INK_MUTED, fontsize=8)


def grid_edges(low: float, high: float, bins: int, log_scale: bool) -> list[float]:
    """Common bin edges for a resampled marginal, geometric when the prior is log-scaled."""
    if log_scale and low > 0.0:
        ratio = (high / low) ** (1.0 / bins)
        return [low * (ratio**index) for index in range(bins + 1)]
    step = (high - low) / bins
    return [low + (step * index) for index in range(bins + 1)]


def add_point_mass(edges: Sequence[float], mass: list[float], value: float, amount: float) -> None:
    for index in range(len(mass)):
        if edges[index] <= value <= edges[index + 1]:
            mass[index] += amount
            return
    mass[0 if value < edges[0] else -1] += amount


def add_interval_mass(edges: Sequence[float], mass: list[float], low: float, high: float, amount: float) -> None:
    """Spread one source bin's mass across the common grid in proportion to overlap."""
    if high <= low:
        add_point_mass(edges, mass, low, amount)
        return
    for index in range(len(mass)):
        overlap = min(high, edges[index + 1]) - max(low, edges[index])
        if overlap > 0.0:
            mass[index] += amount * (overlap / (high - low))


def component_support(spec: Mapping[str, Any]) -> tuple[float, float]:
    if distribution_kind(spec) == "choice":
        values = [float(value) for value, _ in choice_values(spec)]
        return min(values), max(values)
    edges = spec.get("edges")
    if edges:
        return float(edges[0]), float(edges[-1])
    return float(spec["low"]), float(spec["high"])


def plot_choice(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Bar chart of normalized probability over an explicit, enumerable support."""
    pairs = choice_values(spec)
    total = sum(weight for _, weight in pairs) or 1.0
    positions = list(range(len(pairs)))
    axes.bar(positions, [weight / total for _, weight in pairs], width=0.62, color=SERIES)
    axes.set_xticks(positions)
    axes.set_xticklabels([format_value(value) for value, _ in pairs])
    axes.set_xlim(-0.6, len(pairs) - 0.4)
    axes.set_ylim(0.0, 1.0)
    axes.set_ylabel("probability", color=INK_MUTED, fontsize=8)


def plot_uniform(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Exact flat density over [low, high], with any declared `samples` landmarks marked."""
    low = float(spec["low"])
    high = float(spec["high"])
    span = high - low
    density = 1.0 / span if span > 0.0 else 0.0
    landmarks = [float(value) for value in spec.get("samples", [])]
    axes.fill_between([low, high], 0.0, density, color=SERIES, alpha=0.22, linewidth=0.0)
    axes.plot([low, high], [density, density], color=SERIES, linewidth=2.0)
    if landmarks:
        axes.vlines(landmarks, 0.0, density, color=INK_MUTED, linewidth=1.0, linestyles=(0, (3, 3)))
    ticks = landmarks or [low, high]
    axes.set_xticks(ticks)
    axes.set_xticklabels([format_value(value) for value in ticks])
    axes.set_xlim(low - (span * 0.08 or 0.5), high + (span * 0.08 or 0.5))
    axes.set_ylim(0.0, density * 1.35 if density > 0.0 else 1.0)
    axes.set_ylabel("density", color=INK_MUTED, fontsize=8)


def plot_histogram(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Empirical bins exactly as declared: bar i spans edges[i]..edges[i+1] at weights[i]."""
    edges = [float(value) for value in spec["edges"]]
    draw_bins(axes, edges, [float(value) for value in spec["weights"]], spec.get("scale") == "log")


def plot_normal(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Normal density truncated to the observed [low, high], with the mean marked."""
    low = float(spec["low"])
    high = float(spec["high"])
    mean = float(spec["mean"])
    std = float(spec["std"]) or 1.0
    xs = [low + ((high - low) * index / 200.0) for index in range(201)]
    ys = [math.exp(-0.5 * (((x - mean) / std) ** 2)) / (std * math.sqrt(2.0 * math.pi)) for x in xs]
    axes.fill_between(xs, 0.0, ys, color=SERIES, alpha=0.22, linewidth=0.0)
    axes.plot(xs, ys, color=SERIES, linewidth=2.0)
    axes.vlines([mean], 0.0, max(ys), color=INK_MUTED, linewidth=1.0, linestyles=(0, (3, 3)))
    axes.annotate("mean %s" % format_value(mean), xy=(mean, max(ys)), xytext=(0, 3), textcoords="offset points", ha="center", color=INK_MUTED, fontsize=7)
    axes.set_xticks([low, mean, high])
    axes.set_xticklabels([format_value(value) for value in (low, mean, high)])
    axes.set_xlim(low, high)
    axes.set_ylim(0.0, max(ys) * 1.35)
    axes.set_ylabel("density", color=INK_MUTED, fontsize=8)


def plot_joint(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Weighted mixture marginal for one parameter of a joint, resampled onto a common grid."""
    parts = spec["parts"]
    note = "marginal of %d-component mixture" % len(parts)
    if all(distribution_kind(part) == "choice" for _, part in parts):
        # An all-discrete mixture is still a discrete support: merge it and draw real bars
        # rather than scattering two point masses across a continuous grid.
        merged: dict[float, float] = {}
        for weight, part in parts:
            pairs = choice_values(part)
            total = sum(item for _, item in pairs) or 1.0
            for value, item in pairs:
                merged[float(value)] = merged.get(float(value), 0.0) + (weight * item / total)
        plot_choice(axes, {"type": "choice", "values": [{"value": value, "weight": item} for value, item in sorted(merged.items())]})
        axes.annotate(note, xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", color=INK_MUTED, fontsize=7)
        return
    supports = [component_support(part) for _, part in parts]
    low = min(bound[0] for bound in supports)
    high = max(bound[1] for bound in supports)
    if high <= low:
        high = low + 1.0
    log_scale = low > 0.0 and any(part.get("scale") == "log" for _, part in parts)
    edges = grid_edges(low, high, MARGINAL_BINS, log_scale)
    mass = [0.0] * MARGINAL_BINS
    for weight, part in parts:
        if distribution_kind(part) == "choice":
            pairs = choice_values(part)
            total = sum(item for _, item in pairs) or 1.0
            for value, item in pairs:
                add_point_mass(edges, mass, float(value), weight * (item / total))
            continue
        if "edges" in part:
            part_edges = [float(value) for value in part["edges"]]
            for index, bin_weight in enumerate(part["weights"]):
                add_interval_mass(edges, mass, part_edges[index], part_edges[index + 1], weight * float(bin_weight))
            continue
        add_interval_mass(edges, mass, float(part["low"]), float(part["high"]), weight)
    draw_bins(axes, edges, mass, log_scale)
    axes.annotate(note, xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", color=INK_MUTED, fontsize=7)


def plot_unsupported(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Placeholder so a kind this script does not know stays visible instead of vanishing."""
    axes.text(0.5, 0.5, "unsupported type\n%s" % distribution_kind(spec), ha="center", va="center", color=INK_MUTED, fontsize=9, transform=axes.transAxes)
    axes.set_xticks([])
    axes.set_yticks([])


# Kind -> panel renderer. The choice aliases mirror DISTRIBUTION_HANDLER_NAMES
# (ground_truth/planner.py) and SUPPORT_HANDLER_NAMES (ground_truth/abstraction.py).
RENDERERS = {
    "choice": plot_choice,
    "grid": plot_choice,
    "values": plot_choice,
    "uniform": plot_uniform,
    "int_uniform": plot_uniform,
    "histogram": plot_histogram,
    "normal": plot_normal,
    "joint": plot_joint
}


def plot_effect(effect_path: str, leaves: list[tuple[str, Mapping[str, Any]]], output_dir: Path) -> Path:
    """Render one effect's parameters as a grid of subplots and write it as a single PDF."""
    count = len(leaves)
    columns = count if count <= 3 else math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    figure, grid = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows), squeeze=False, layout="constrained")
    for index, axes in enumerate(grid.flat):
        if index >= count:
            axes.set_axis_off()
            continue
        name, spec = leaves[index]
        kind = distribution_kind(spec)
        if kind not in RENDERERS:
            stage("  unsupported type '%s' for %s.%s" % (kind, effect_path, name))
        RENDERERS.get(kind, plot_unsupported)(axes, spec)
        style_axes(axes)
        axes.set_title("%s: %s" % (name, kind), color=INK, fontsize=10, pad=8)
    figure.suptitle(effect_path, color=INK, fontsize=12)
    path = output_dir / ("%s.pdf" % effect_path.replace(".", "_"))
    # Pin CreationDate so repeated runs are byte-identical and the output stays diffable.
    figure.savefig(path, metadata={"CreationDate": None})
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="Ground-truth config directory. Default: %(default)s")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for the per-effect PDF panels. Default: %(default)s")
    args = parser.parse_args()

    with (args.config_dir / "distributions.yaml").open("r", encoding="utf-8") as handle:
        distributions = (yaml.safe_load(handle) or {}).get("distributions", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    panels = 0
    for effect_path, node in effect_nodes(distributions):
        leaves = [(name or effect_path.split(".")[-1], spec) for name, spec in collect_leaves(node)]
        if not leaves:
            stage("%s has no distributions, skipping" % effect_path)
            continue
        path = plot_effect(effect_path, leaves, args.output_dir)
        total += len(leaves)
        panels += 1
        stage("%s -> %s (%d parameters)" % (effect_path, path.name, len(leaves)))
    stage("wrote %d parameters across %d panels to %s" % (total, panels, args.output_dir))


if __name__ == "__main__":
    main()
