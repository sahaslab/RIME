import os
import json
import argparse
import hashlib
import html
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from tqdm import tqdm
from plan_audit_utils import (
    load_configuration,
    load_analysis,
    extract_parameters,
    discrete_probabilities,
    category_index,
    support_bounds,
    in_support,
    evaluate_cdf,
    coverage_bins,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument(
        "--priors-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "generation_priors",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--analysis-path", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plan_audit",
    )
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()
    assert args.bins > 0
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rime-audit-matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    config_dir = args.config_dir or args.priors_dir / "03_rime"
    routes, priors, models = load_configuration(config_dir)
    analysis = load_analysis(args.analysis_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, unmatched = [], []
    grouped = defaultdict(list)
    references = {("graph_spec", name): model for name, model in models.items()}
    plan_count = 0
    with args.plans.open() as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc="Auditing plan parameters"), start=1
        ):
            if not line.strip():
                continue
            plan = json.loads(line)
            plan_count += 1
            identity = {
                "line": line_number,
                "plan_id": plan.get("plan_id"),
                "clip_id": plan.get("clip_id"),
                "recipe_id": plan["recipe_id"],
            }
            rows, missing = extract_parameters(
                plan,
                routes,
                priors,
                plan.get("analysis", analysis.get(plan["clip_id"])),
            )
            unmatched.extend(identity | item for item in missing)
            for row in rows:
                model = row.pop("model")
                row["in_support"] = (
                    in_support(model, row["value"]) if model is not None else None
                )
                row.update(identity)
                samples.append(row)
                key = (row["graph"], row["prior"])
                grouped[key].append(row)
                # Context-dependent settings have per-row expectations, not one pooled prior.
                references[key] = None if row["status"] == "context" else model
                if model is None:
                    unmatched.append(
                        identity
                        | {
                            "graph": row["graph"],
                            "prior": row["prior"],
                            "reason": row["status"],
                        }
                    )
    columns = [
        "line",
        "plan_id",
        "clip_id",
        "recipe_id",
        "graph",
        "node",
        "step",
        "operator",
        "parameter",
        "prior",
        "value",
        "resolved_value",
        "status",
        "in_support",
    ]
    pd.DataFrame(samples, columns=columns).to_csv(
        args.output_dir / "values.csv", index=False
    )
    pd.DataFrame(
        unmatched,
        columns=[
            "line",
            "plan_id",
            "clip_id",
            "recipe_id",
            "graph",
            "prior",
            "node",
            "step",
            "parameter",
            "reason",
        ],
    ).to_csv(args.output_dir / "unmatched.csv", index=False)
    summaries, sections = [], []
    with PdfPages(args.output_dir / "plan_audit.pdf") as pdf:
        for (graph, name), model in tqdm(
            references.items(), desc="Plotting parameter audit"
        ):
            rows = grouped[(graph, name)]
            summary = summarize(rows, model, args.bins) | {
                "graph": graph,
                "prior": name,
            }
            summaries.append(summary)
            if not rows:
                continue
            fig = plot_parameter(rows, model, args.bins, plt)
            fig.suptitle(
                "%s | %s\nn=%d; outside support=%d"
                % (graph, name, len(rows), summary["out_of_support"]),
                fontsize=10,
            )
            fig.tight_layout()
            filename = hashlib.sha256((graph + name).encode()).hexdigest()[:16] + ".png"
            fig.savefig(args.output_dir / filename, dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
            sections.append(
                '<h2>%s: %s</h2><img src="%s">'
                % (html.escape(graph), html.escape(name), filename)
            )
    table = pd.DataFrame(summaries)
    table.to_csv(args.output_dir / "summary.csv", index=False)
    paths = [args.plans] + [
        config_dir / name
        for name in ("distributions.yaml", "recipes.yaml", "motifs.yaml")
    ]
    if args.analysis_path:
        paths.append(args.analysis_path)
    hashes = {}
    for path in tqdm(paths, desc="Hashing audit inputs"):
        with path.open("rb") as handle:
            hashes[str(path.resolve())] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
    (args.output_dir / "inputs.json").write_text(json.dumps(hashes, indent=2) + "\n")
    notes = (
        "<p>%d plans; %d parameter occurrences; %d unresolved or missing references.</p><p>Corrective graphs, poison graphs, and scalar bindings are reported separately. Continuous priors use CDF distance and quantile-bin coverage; discrete priors use category coverage and total variation distance. Fixed controls and context-dependent values are checked against their configured values. Unresolved parameters remain in the inventory and plots, with reasons in unmatched.csv. Values stores both resolved graph values and recovered prior-space values.</p><p>All configured scalar priors appear in the corrective-graph table, including priors with no observations there. Use the configuration and analysis manifest that generated these plans. Legacy metadata-dependent expressions require --analysis-path; missing metadata is not replaced by an assumed fallback. Coverage measures occupied bins, not independent draws. Selection, enumeration, and repeated or reused values affect distribution comparisons. No p-values are computed.</p>"
        % (plan_count, len(samples), len(unmatched))
    )
    (args.output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Plan audit</title><style>body{font:15px system-ui;max-width:1250px;margin:30px auto}img{width:100%}td,th{padding:6px}</style><h1>Plan audit</h1>'
        + notes
        + '<p><a href="plan_audit.pdf">PDF</a> · <a href="summary.csv">Summary</a> · <a href="values.csv">Values</a> · <a href="unmatched.csv">Unmatched</a></p>'
        + table.to_html(index=False)
        + "".join(sections)
    )
    print(
        "Audited %d plans and %d parameter occurrences. Output: %s"
        % (plan_count, len(samples), args.output_dir)
    )


def summarize(
    rows: list[dict[str, Any]],
    model: dict[str, Any] | None,
    bins: int,
) -> dict[str, Any]:
    values = [row["value"] for row in rows]
    summary = {
        "n": len(values),
        "distinct_values": len({json.dumps(value, sort_keys=True) for value in values}),
        "unique_clips": len({row["clip_id"] for row in rows}),
        "out_of_support": sum(row["in_support"] is False for row in rows),
        "unresolved": sum(row["in_support"] is None for row in rows),
        "cdf_distance": None,
        "total_variation": None,
        "occupied_bins": None,
        "total_bins": None,
        "coverage": None,
    }
    if model is None:
        summary["kind"] = "observed"
        return summary
    categories = discrete_probabilities(model)
    summary["kind"] = "discrete" if categories is not None else "continuous"
    assignments = coverage_bins(model, values, bins)
    summary["occupied_bins"] = len(
        {bucket for bucket in assignments if bucket is not None}
    )
    summary["total_bins"] = (
        sum(probability > 0 for _, probability in categories)
        if categories is not None
        else bins
    )
    summary["coverage"] = summary["occupied_bins"] / summary["total_bins"]
    if not values:
        return summary
    if categories is not None:
        counts = Counter(category_index(value, categories) for value in values)
        summary["total_variation"] = float(
            (
                sum(
                    abs(counts[index] / len(values) - probability)
                    for index, (_, probability) in enumerate(categories)
                )
                + counts[-1] / len(values)
            )
            / 2
        )
    else:
        finite = np.sort(
            [
                value
                for value in values
                if isinstance(value, (int, float)) and np.isfinite(value)
            ]
        )
        if len(finite):
            cdf = evaluate_cdf(model, finite)
            summary["cdf_distance"] = float(
                max(
                    np.max(np.arange(1, len(finite) + 1) / len(finite) - cdf),
                    np.max(cdf - np.arange(len(finite)) / len(finite)),
                )
            )
    return summary


def plot_parameter(
    rows: list[dict[str, Any]],
    model: dict[str, Any] | None,
    bins: int,
    plt: Any,
) -> Any:
    values = [row["value"] for row in rows]
    categories = discrete_probabilities(model) if model is not None else None
    if categories is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        counts = Counter(category_index(value, categories) for value in values)
        labels = [str(value) for value, _ in categories] + ["outside support"]
        expected = [probability for _, probability in categories] + [0.0]
        observed = [counts[index] / len(values) for index in range(len(categories))] + [
            counts[-1] / len(values)
        ]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, observed, width=0.4, label="Plans")
        ax.bar(x + 0.2, expected, width=0.4, label="Configured probabilities")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel("Probability")
        ax.legend()
        return fig
    if model is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        if all(isinstance(value, (int, float)) for value in values):
            ax.hist([value for value in values if np.isfinite(value)], bins=bins)
        else:
            counts = Counter(str(value) for value in values)
            ax.bar(list(counts), list(counts.values()))
            ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Observed occurrences")
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    low, high = support_bounds(model)
    logarithmic = model.get("scale") == "log" or model["type"] in {
        "log_uniform",
        "power_law",
    }
    if model["type"] == "marginal_mixture":
        logarithmic = all(
            part["model"].get("scale") == "log" for part in model["components"]
        )
    coordinate = (
        np.linspace(np.log(low), np.log(high), 2001)
        if logarithmic
        else np.linspace(low, high, 2001)
    )
    x = np.exp(coordinate) if logarithmic else coordinate
    edges = np.linspace(coordinate[0], coordinate[-1], bins + 1)
    finite = np.sort(
        [
            value
            for value in values
            if isinstance(value, (int, float)) and np.isfinite(value)
        ]
    )
    valid = finite[(finite >= low) & (finite <= high)]
    mass, _ = np.histogram(np.log(valid) if logarithmic else valid, bins=edges)
    axes[0].stairs(
        mass / len(values) / np.diff(edges),
        np.exp(edges) if logarithmic else edges,
        fill=True,
        alpha=0.35,
        label="Plans",
    )
    curve = evaluate_cdf(model, x)
    axes[0].plot(
        (x[:-1] + x[1:]) / 2,
        np.diff(curve) / np.diff(coordinate),
        label="Configured prior",
    )
    if len(finite):
        axes[1].step(
            finite,
            np.arange(1, len(finite) + 1) / len(finite),
            where="post",
            label="Plan CDF",
        )
    axes[1].plot(x, curve, label="Prior CDF")
    axes[0].set_ylabel("Density per log unit" if logarithmic else "Density")
    axes[1].set_ylabel("Cumulative probability")
    for ax in axes:
        if logarithmic:
            ax.set_xscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(bottom=0)
        ax.legend()
    return fig


if __name__ == "__main__":
    main()
