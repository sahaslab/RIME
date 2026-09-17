"""Plot the sampling priors in distributions.yaml, one PDF panel per effect.

distributions.yaml is a nested tree grouped by workflow family rather than by effect, so the
unit plotted here is the level-2 `<family>.<effect>` node (e.g. `vocals.compression`). Every
leaf distribution beneath it becomes one subplot titled "<parameter>: <type>", and the panel is
written to <output-dir>/<family>_<effect>.pdf.

Rendering is fully deterministic: choice supports are drawn from their declared weights and
uniform ranges from their closed-form density, so no RNG is involved and the PDFs are
byte-reproducible across runs.
"""

import math
import argparse
from pathlib import Path
from collections.abc import Mapping
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
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"


def stage(message: str) -> None:
    tqdm.write(f"[plot-distributions] {message}")


def format_value(value: Any) -> str:
    """Trim trailing zeros on floats so ticks read 3800 and -3 rather than 3800.0 and -3.0."""
    return "%g" % value if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


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


def collect_leaves(node: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    """(name relative to the effect node, leaf spec) pairs, in file order."""
    if is_leaf(node):
        return [(prefix, node)]
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
    """Exact flat density over [low, high], with the declared `samples` landmarks marked."""
    low = float(spec["low"])
    high = float(spec["high"])
    span = high - low
    density = 1.0 / span if span > 0.0 else 0.0
    landmarks = [float(value) for value in spec.get("samples", [low, high])]
    axes.fill_between([low, high], 0.0, density, color=SERIES, alpha=0.22, linewidth=0.0)
    axes.plot([low, high], [density, density], color=SERIES, linewidth=2.0)
    axes.vlines(landmarks, 0.0, density, color=INK_MUTED, linewidth=1.0, linestyles=(0, (3, 3)))
    axes.set_xticks(landmarks)
    axes.set_xticklabels([format_value(value) for value in landmarks])
    axes.set_xlim(low - (span * 0.08 or 0.5), high + (span * 0.08 or 0.5))
    axes.set_ylim(0.0, density * 1.35 if density > 0.0 else 1.0)
    axes.set_ylabel("density", color=INK_MUTED, fontsize=8)


def plot_unsupported(axes: Axes, spec: Mapping[str, Any]) -> None:
    """Placeholder so a kind this script does not know stays visible instead of vanishing."""
    axes.text(0.5, 0.5, "unsupported type\n%s" % distribution_kind(spec), ha="center", va="center", color=INK_MUTED, fontsize=9, transform=axes.transAxes)
    axes.set_xticks([])
    axes.set_yticks([])


# Kind -> panel renderer. Aliases mirror DISTRIBUTION_HANDLER_NAMES (ground_truth/planner.py)
# and SUPPORT_HANDLER_NAMES (ground_truth/abstraction.py); keep the three tables in step.
RENDERERS = {
    "choice": plot_choice,
    "grid": plot_choice,
    "values": plot_choice,
    "uniform": plot_uniform,
    "int_uniform": plot_uniform
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
    for effect_path, node in effect_nodes(distributions):
        leaves = [(name or effect_path.split(".")[-1], spec) for name, spec in collect_leaves(node)]
        if not leaves:
            stage("%s has no distributions, skipping" % effect_path)
            continue
        path = plot_effect(effect_path, leaves, args.output_dir)
        total += len(leaves)
        stage("%s -> %s (%d parameters)" % (effect_path, path.name, len(leaves)))
    stage("wrote %d parameters across %d panels to %s" % (total, len(effect_nodes(distributions)), args.output_dir))


if __name__ == "__main__":
    main()
