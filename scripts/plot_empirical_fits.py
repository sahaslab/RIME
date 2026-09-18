import os
import json
import argparse
import html
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from collections import defaultdict
from typing import Any
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors-dir", type=Path, default=Path(__file__).resolve().parents[1] / "generation_priors")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "empirical")
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()
    assert args.bins > 0
    import extract_priors as extraction
    from build_corpus import digest
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rime-empirical-matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    priors = json.loads((args.priors_dir / "02_priors/priors.json").read_text())
    presets_path = args.priors_dir / "02_priors/presets.jsonl"
    assert digest(presets_path) == priors["presets_sha256"]
    assert all(digest(Path(__file__).resolve().parent / name) == expected for name, expected in priors["scripts_sha256"].items())
    inventory = json.loads((args.priors_dir / "01_corpus/inventory.json").read_text())
    assert digest(args.priors_dir / "01_corpus/inventory.json") == priors["input_sha256"]
    records = [json.loads(line) for line in presets_path.read_text().splitlines()]
    records = [row for row in records if priors["splits"] is None or row["split"] in priors["splits"]]
    observations, _ = extraction.translate(records, inventory, extraction.MAPPINGS)
    urls = {asset["path"]: asset["url"] for asset in inventory["assets"]}
    items = plot_items(priors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections, evidence, checks = [], [], []
    with PdfPages(args.output_dir / "empirical_fits.pdf") as pdf:
        for name, item in tqdm(items.items(), desc="Comparing empirical data with fits"):
            model, report = item["model"], item["report"]
            rows = empirical_rows(observations, item, urls, extraction)
            values = np.array([row["value"] for row in rows])
            weights = extraction.observation_weights(rows)
            assert len(rows) == report["observations"] - report["outside_domain"], name
            assert len({row["record_id"] for row in rows}) == report["records"], name
            assert extraction.upstream_provenance(rows) == report["provenance"], name
            context_mass = defaultdict(float)
            for row, weight in zip(rows, weights):
                context_mass[(row["source"], row["context"].get("song") or row["record_id"])] += float(weight)
                evidence.append({"prior": name, "value": row["value"], "weight": float(weight), "source": row["source"], "record_id": row["record_id"], "instance_id": row["instance_id"], "context": row["context"].get("song") or row["record_id"], "source_url": urls[row["asset"]], "source_row_index": row["row_index"]})
            effective = 1 / sum(mass ** 2 for mass in context_mass.values())
            assert np.isclose(effective, report["effective_contexts"]), name
            assert len(context_mass) == report["contexts"], name
            assert np.allclose([values.min(), values.max()], report["observed_range"]), name
            logarithmic = model.get("scale") == "log"
            low, high = model["low"], model["high"]
            a, b = (np.log(low), np.log(high)) if logarithmic else (low, high)
            edges = np.linspace(a, b, args.bins + 1)
            coordinate = np.linspace(a, b, 4001)
            x = np.exp(coordinate) if logarithmic else coordinate
            display_edges = np.exp(edges) if logarithmic else edges
            empirical_mass, _ = np.histogram(np.log(values) if logarithmic else values, bins=edges, weights=weights)
            fitted_cdf = extraction.model_cdf(model, x)
            fitted_mass = np.diff(extraction.model_cdf(model, display_edges))
            order = np.argsort(values)
            sorted_values, sorted_weights = values[order], weights[order]
            unique, starts = np.unique(sorted_values, return_index=True)
            empirical_cdf = np.cumsum(np.add.reduceat(sorted_weights, starts))
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            axes[0].stairs(empirical_mass / np.diff(edges), display_edges, fill=True, alpha=0.3, color="#df8b38", label="Weighted empirical data")
            axes[0].plot((x[:-1] + x[1:]) / 2, np.diff(fitted_cdf) / np.diff(coordinate), color="#24658d", linewidth=2, label=family_label(model))
            axes[0].stairs(fitted_mass / np.diff(edges), display_edges, color="#24658d", linestyle=":", linewidth=1, label="Fit averaged within bins")
            axes[1].step(np.r_[low, unique, high], np.r_[0, empirical_cdf, 1], where="post", color="#c77322", linewidth=1.7, label="Weighted empirical CDF")
            axes[1].plot(x, fitted_cdf, color="#24658d", linewidth=2, label="Fitted CDF")
            axes[0].set_ylabel("Density per natural-log unit" if logarithmic else "Density per parameter unit")
            axes[1].set_ylabel("Cumulative probability")
            axes[1].set_ylim(0, 1.02)
            for ax in axes:
                if logarithmic:
                    ax.set_xscale("log")
                ax.set_xlim(low, high)
                ax.set_ylim(bottom=0)
                ax.set_xlabel(item["parameter"] + (" (log axis)" if logarithmic else ""))
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(alpha=0.15)
                ax.legend(fontsize=8)
            fig.suptitle("%s\n%s | %d observations | %d records | %.1f effective contexts" % (name, family_label(model), len(rows), report["records"], effective), fontsize=11)
            fig.tight_layout()
            filename = name.replace(".", "_") + ".png"
            fig.savefig(args.output_dir / filename, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            checks.append({"prior": name, "family": family_label(model), "observations": len(rows), "records": report["records"], "effective_contexts": effective, "weight_sum": float(weights.sum())})
            sections.append("<section><h2>%s</h2><img src=\"%s\" alt=\"Empirical data and fit for %s\"><pre>%s</pre></section>" % (html.escape(name), filename, html.escape(name), html.escape(json.dumps(model, indent=2))))
    pd.DataFrame(evidence).to_csv(args.output_dir / "empirical_observations.csv", index=False)
    pd.DataFrame(checks).to_csv(args.output_dir / "checks.csv", index=False)
    page = """<!doctype html><meta charset="utf-8"><title>Empirical data against fitted priors</title><style>body{font:15px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:20px;color:#17232f}section{border-top:1px solid #ddd;margin-top:30px;padding-top:15px}img{width:100%%}h2{font-size:17px;overflow-wrap:anywhere}pre{font-size:12px;white-space:pre-wrap}</style><h1>Empirical data against fitted priors</h1><p>Orange: observed settings with the same source/context weights used for fitting. Blue: the saved fitted distribution. Histograms use %d equally spaced bins in the fitting coordinate; dotted blue steps show fitted mass averaged over those same bins. The empirical CDF has no binning. Repeated observations are retained with their original fitting weights.</p><p>These comparisons use the accepted training observations, after the production filters and complete-setting requirements; they are not held-out validation. Record counts, original source rows, context counts, effective context counts, and observed ranges are checked against the saved fit reports. Each high-shelf sign branch is normalized separately.</p><p>Frequency, Q, and ratio plots use a log axis and density per natural-log unit. A normal fitted in log space is labeled bounded log-normal; a Gaussian mixture in log space is labeled a mixture of bounded log-normals. Signed dB controls remain linear. Changing the histogram bin count does not change the fits.</p><p><a href="empirical_fits.pdf">All plots as PDF</a> · <a href="empirical_observations.csv">Exact plotted observations and weights</a> · <a href="checks.csv">Reconstruction checks</a> · <a href="../plot_empirical_fits.py">Script</a></p>""" % args.bins
    (args.output_dir / "index.html").write_text(page + "".join(sections))
    print("Plotted %d empirical comparisons in %s" % (len(items), args.output_dir))


def plot_items(priors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = {}
    references = {value["sample"] for model in priors["distributions"].values() if model["type"] == "parameters" for value in model["parameters"].values() if "sample" in value}
    for path, parent in priors["distributions"].items():
        if path in references:
            continue
        report = priors["fit_reports"][path]
        operator = "apply_compressor_effect" if "compression" in path else "apply_%s_filter" % path.split(".")[2]
        if parent["type"] in {"parameters", "mixture"}:
            components = parent["components"] if parent["type"] == "mixture" else [{"label": "all", "distribution": parent}]
            for component, partition in zip(components, report["partitions"]):
                for parameter, model in component["distribution"]["parameters"].items():
                    name = model.get("sample", path + "." + component["label"] + "." + parameter)
                    model = priors["distributions"][model["sample"]] if "sample" in model else model
                    items[name] = {"model": model, "report": partition["fits"][parameter], "operator": operator, "parameter": parameter, "controls": report["controls"], "branch": component["label"]}
        else:
            items[path] = {"model": parent, "report": report, "operator": operator, "parameter": path.split(".")[-1], "controls": None, "branch": "all"}
    return items


def empirical_rows(
    observations: list[dict[str, Any]],
    item: dict[str, Any],
    urls: dict[str, str],
    extraction: Any
) -> list[dict[str, Any]]:
    provenance = {(entry["url"], index) for entry in item["report"]["provenance"] for index in entry["rows"]}
    selected = [row for row in observations if row["operator"] == item["operator"] and (urls[row["asset"]], row["row_index"]) in provenance]
    policy = extraction.FIT_POLICY
    selected = [row for row in selected if row["parameter"] in policy["domains"] and row["parameter"] not in policy["blocked_controls"].get(row["source"], []) and row["routing"] != "send" and policy["domains"][row["parameter"]][0] <= row["value"] <= policy["domains"][row["parameter"]][1]]
    if item["controls"] is None:
        return [row for row in selected if row["parameter"] == item["parameter"] and item["model"]["low"] <= row["value"] <= item["model"]["high"]]
    settings, lookup = defaultdict(dict), {}
    for row in selected:
        if row["parameter"] in item["controls"]:
            key = (row["source"], row["record_id"], row["instance_id"])
            settings[key][row["parameter"]] = row["value"]
            lookup[key] = row
    result = []
    for key in sorted(settings):
        values = settings[key]
        if set(values) != set(item["controls"]):
            continue
        if item["operator"] == "apply_highshelf_filter":
            low, high = policy["highshelf_gain_magnitude_db"]
            gain = values["gain_db"]
            if not low <= abs(gain) <= high or (gain < 0) != (item["branch"] == "cut"):
                continue
        row = lookup[key]
        context = row["context"].copy()
        if not context.get("song"):
            context["song"] = "setting:" + extraction.stable_id(values)
        result.append(row | {"value": values[item["parameter"]], "context": context})
    return result


def family_label(model: dict[str, Any]) -> str:
    if model["type"] == "normal":
        return "Bounded log-normal" if model.get("scale") == "log" else "Bounded normal"
    if model["type"] == "gaussian_mixture":
        return "Two bounded log-normals" if model.get("scale") == "log" else "Two bounded normals"
    return model["type"].replace("_", " ") + (" in log coordinates" if model["type"] == "beta" and model.get("scale") == "log" else "")


if __name__ == "__main__":
    main()
