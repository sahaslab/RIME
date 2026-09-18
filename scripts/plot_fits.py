import os
import json
import argparse
import html
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors-dir", type=Path, default=Path(__file__).resolve().parents[1] / "generation_priors")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    from extract_priors import model_cdf
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rime-mle-matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    args.output_dir.mkdir(parents=True, exist_ok=True)
    priors = json.loads((args.priors_dir / "02_priors/priors.json").read_text())
    plots, scores, correlations = [], [], []
    with PdfPages(args.output_dir / "fits.pdf") as pdf:
        for path, parent in tqdm(priors["distributions"].items(), desc="Plotting MLE fits"):
            report = priors["fit_reports"][path]
            if parent["type"] in {"parameters", "mixture"}:
                components = parent["components"] if parent["type"] == "mixture" else [{"label": "all", "distribution": parent}]
                items = []
                for component, partition in zip(components, report["partitions"]):
                    correlations.extend({"prior": path, "branch": partition["label"]} | item for item in partition["correlations"])
                    for parameter, model in component["distribution"]["parameters"].items():
                        if "sample" not in model:
                            items.append((path + "." + component["label"] + "." + parameter, model, partition["fits"][parameter]))
            else:
                items = [(path, parent, report)]
            for name, model, fit in items:
                logarithmic = model.get("scale") == "log"
                low, high = model["low"], model["high"]
                coordinate = np.linspace(np.log(low), np.log(high), 3001) if logarithmic else np.linspace(low, high, 3001)
                x = np.exp(coordinate) if logarithmic else coordinate
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                cdf = model_cdf(model, x)
                axes[0].plot((x[:-1] + x[1:]) / 2, np.diff(cdf) / np.diff(coordinate), color="#24658d", linewidth=2)
                axes[1].plot(x, cdf, color="#24658d", linewidth=2)
                axes[0].set_ylabel("Density per natural-log unit" if logarithmic else "Density per parameter unit")
                axes[1].set_ylabel("Cumulative probability")
                for ax in axes:
                    if logarithmic:
                        ax.set_xscale("log")
                    ax.set_xlim(low, high)
                    ax.set_ylim(bottom=0)
                    ax.set_xlabel(name.split(".")[-1] + (" (log x axis)" if logarithmic else ""))
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.grid(alpha=0.15)
                fig.suptitle("%s\n%s | %d records | %.1f effective contexts" % (name, model["type"], fit["records"], fit["effective_contexts"]), fontsize=12)
                fig.tight_layout()
                filename = name.replace(".", "_") + ".png"
                fig.savefig(args.output_dir / filename, dpi=140)
                pdf.savefig(fig)
                plt.close(fig)
                table = []
                for candidate in fit["candidates"]:
                    family = candidate["model"]["type"] if "model" in candidate else candidate["family"]
                    row = {"prior": name, "family": family, "bic_score": candidate.get("bic_score"), "parameters": candidate.get("parameters"), "at_bound": candidate.get("at_numerical_bound", False), "excluded": candidate.get("excluded", ""), "selected": "model" in candidate and candidate["model"] == model}
                    scores.append(row)
                    table.append(row | {"prior": ""})
                plots.append("<section><h2>%s</h2><img src=\"%s\" alt=\"%s\"><pre>%s</pre>%s</section>" % (html.escape(name), filename, html.escape(name), html.escape(json.dumps(model, indent=2)), pd.DataFrame(table).drop(columns="prior").to_html(index=False, float_format=lambda value: "%.3f" % value)))
    pd.DataFrame(scores).to_csv(args.output_dir / "model_scores.csv", index=False)
    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(args.output_dir / "correlations.csv", index=False)
    notes = """<p>Current production fits: weighted maximum likelihood, fixed admissible bounds, and a small candidate family list. These are sampling distributions, not Bayesian posteriors. No histogram fits or numerical cluster profiles are generated.</p><p>Selection uses a BIC-style score: −2 N_eff Σ w_i log f(x_i) + k log N_eff. Weights balance sources and contexts; N_eff is the inverse sum of squared context masses. This is a clustered-data approximation, not standard IID BIC or a Bayes factor. Scores are comparable within each parameter only. The likelihood is evaluated in normalized linear/log coordinates.</p><p>Normal component centers stay inside the allowed range, and the distributions are truncated to those bounds. Minimum normal spread is 2% of the transformed allowed range; a mixture has two components with at least 5% weight each. These declared safeguards prevent degenerate mixture fits. Numerical parameter bounds and boundary hits are recorded in the fitting code and diagnostics. Beta fitting is skipped when an observation equals a support endpoint; we do not silently move observations to make its point-density likelihood finite. Power-law density is proportional to x raised to the printed exponent over the positive finite range.</p><p>Parameter sets are independent. High-shelf cut/boost branches are explicit sign categories, preserving the forbidden gain gap. The correlation screen requires both Pearson and Spearman magnitudes ≥0.8 with matching sign; no within-source pair passed in this corpus, so no correlated joint model was fitted. We have not claimed that this threshold establishes independence.</p><p>Source references and original datapoint locations remain in the generated YAML comments and priors.json. Existing authored controls retain their declared defaults; this review covers fitted marginals.</p><p><a href="fits.pdf">All curves as PDF</a> · <a href="model_scores.csv">Candidate scores</a> · <a href="correlations.csv">Correlations</a> · <a href="plot_fits.py">Plotting code</a></p>"""
    page = "<!doctype html><meta charset=\"utf-8\"><title>MLE parameter priors</title><style>body{font:15px/1.5 system-ui;max-width:1300px;margin:30px auto;padding:20px;color:#17232f}section{padding:20px;border:1px solid #ddd;margin:25px 0}img{width:100%}h2{font-size:18px;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}pre{font-size:12px;white-space:pre-wrap}</style><h1>MLE parameter priors</h1>" + notes + "<h2>Correlation screen</h2>" + correlation_table.to_html(index=False, float_format=lambda value: "%.3f" % value) + "".join(plots)
    (args.output_dir / "index.html").write_text(page)
    print("Plotted %d fitted marginals." % len(plots))


if __name__ == "__main__":
    main()
