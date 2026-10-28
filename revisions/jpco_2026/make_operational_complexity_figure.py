"""Create the audited replacement Figure 6 and supporting LaTeX table.

Reads only completed files from complexity_benchmark_v1. No models are loaded
and no training, tuning, inference, or benchmark timing is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["XGBoost", "MLP", "FT-Transformer"]
COLORS = {"XGBoost": "#0072B2", "MLP": "#D55E00", "FT-Transformer": "#009E73"}
MARKERS = {"XGBoost": "o", "MLP": "s", "FT-Transformer": "^"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(complexity: pd.DataFrame, latency: pd.DataFrame) -> None:
    assert len(complexity) == 3
    assert set(complexity.model) == set(MODELS)
    assert complexity.training_seeds.eq(3).all()
    assert len(latency) == 12
    assert set(latency.model) == set(MODELS)
    assert set(latency.batch_size) == {1, 32, 256, 5000}
    assert latency.training_seeds.eq(3).all()
    assert not latency.duplicated(["model", "batch_size"]).any()
    required_complexity = [
        "test_weighted_f1_mean", "test_weighted_f1_sd",
        "deployment_bundle_mib_mean", "deployment_bundle_mib_sd",
        "parameter_count_mean", "xgb_boosting_rounds_mean",
        "xgb_tree_count_mean", "xgb_total_nodes_mean", "xgb_total_leaves_mean",
    ]
    required_latency = [
        "latency_median_ms_mean", "latency_median_ms_sd",
        "throughput_rows_per_second_mean", "throughput_rows_per_second_sd",
    ]
    assert np.isfinite(complexity[[c for c in required_complexity[:4]]].to_numpy()).all()
    assert np.isfinite(latency[required_latency].to_numpy()).all()
    assert (complexity.test_weighted_f1_sd >= 0).all()
    assert (complexity.deployment_bundle_mib_mean > 0).all()
    assert (latency.latency_median_ms_mean > 0).all()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_figure(complexity: pd.DataFrame, latency: pd.DataFrame, out: Path) -> None:
    data = complexity.set_index("model").loc[MODELS].copy()
    single = latency[latency.batch_size.eq(1)].set_index("model").loc[MODELS]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharey=True)

    panels = [
        (
            axes[0],
            data.deployment_bundle_mib_mean,
            data.deployment_bundle_mib_sd,
            "Reconstruction-bundle size (MiB)",
            "(a) Serialized footprint",
        ),
        (
            axes[1],
            single.latency_median_ms_mean,
            single.latency_median_ms_sd,
            "Median single-event CPU latency (ms)",
            "(b) Inference latency",
        ),
    ]

    label_offsets = {
        (0, "XGBoost"): (-72, -11),
        (0, "MLP"): (7, 6),
        (0, "FT-Transformer"): (7, -12),
        (1, "XGBoost"): (7, 5),
        (1, "MLP"): (7, 5),
        (1, "FT-Transformer"): (-76, -12),
    }

    for panel_index, (axis, x, xerr, xlabel, title) in enumerate(panels):
        for model in MODELS:
            axis.errorbar(
                x.loc[model],
                data.loc[model, "test_weighted_f1_mean"],
                xerr=xerr.loc[model],
                yerr=data.loc[model, "test_weighted_f1_sd"],
                fmt=MARKERS[model],
                color=COLORS[model],
                markerfacecolor=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=7,
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )
            dx, dy = label_offsets[(panel_index, model)]
            axis.annotate(
                model,
                (x.loc[model], data.loc[model, "test_weighted_f1_mean"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xscale("log")
        axis.set_xlabel(xlabel + "\n(lower is better)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
        axis.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.45, alpha=0.3)
        axis.tick_params(direction="in", top=True, right=True)

    axes[0].set_ylabel("Test weighted F1 score")
    axes[0].set_ylim(0.78, 0.84)
    axes[0].set_yticks(np.arange(0.78, 0.841, 0.01))
    fig.tight_layout(w_pad=1.4)
    fig.savefig(out / "fig6_operational_efficiency.pdf", bbox_inches="tight")
    fig.savefig(out / "fig6_operational_efficiency.png", bbox_inches="tight")
    plt.close(fig)


def fmt(mean: float, sd: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def fmt_size(mean: float, sd: float) -> str:
    if sd < 0.0001:
        return f"{mean:.4f} $\\pm$ $<0.0001$"
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def fmt_count(mean: float, sd: float) -> str:
    return f"{mean:,.0f} $\\pm$ {sd:,.0f}"


def structure(row: pd.Series) -> str:
    if row.name == "XGBoost":
        return (
            f"{int(round(row.xgb_boosting_rounds_mean))} rounds; "
            f"{int(round(row.xgb_tree_count_mean)):,} trees; "
            f"{int(round(row.xgb_total_leaves_mean)):,} leaves"
        )
    return f"{int(round(row.parameter_count_mean)):,} parameters"


def make_table(complexity: pd.DataFrame, latency: pd.DataFrame, out: Path) -> None:
    data = complexity.set_index("model").loc[MODELS]
    single = latency[latency.batch_size.eq(1)].set_index("model").loc[MODELS]
    batch = latency[latency.batch_size.eq(5000)].set_index("model").loc[MODELS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Operational performance and architecture-specific structural descriptors for the three revised pipelines under the grouped-random, nine-feature condition. Values are the mean $\pm$ sample standard deviation across three training seeds. Seed-level latency is the median of repeated CPU measurements. Serialized bundle size and latency are directly measured operational quantities; tree structure and neural-network parameter counts are not treated as equivalent measures of model capacity.}",
        r"\label{tab:operational_complexity}",
        r"\begin{tabular}{lccccc}",
        "Model & Weighted F1 & Bundle size (MiB) & Single-event latency (ms) & Batch throughput (events s$^{-1}$) & Family-specific structure \\\\",
        r"\hline",
    ]
    for model in MODELS:
        row = data.loc[model]
        lines.append(
            f"{model} & "
            f"{fmt(row.test_weighted_f1_mean, row.test_weighted_f1_sd, 4)} & "
            f"{fmt_size(row.deployment_bundle_mib_mean, row.deployment_bundle_mib_sd)} & "
            f"{fmt(single.loc[model].latency_median_ms_mean, single.loc[model].latency_median_ms_sd, 3)} & "
            f"{fmt_count(batch.loc[model].throughput_rows_per_second_mean, batch.loc[model].throughput_rows_per_second_sd)} & "
            f"{structure(row)} \\\\" 
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    (out / "table_operational_complexity.tex").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    source = root / "complexity_benchmark_v1"
    out = root / "complexity_figure_v1"
    out.mkdir(exist_ok=True)

    summary = json.loads((source / "summary.json").read_text())
    assert summary["completed"] is True
    assert summary["training_performed"] is False
    assert summary["workers"] == 9
    assert summary["prediction_replays_passed"] == 9
    complexity = pd.read_csv(source / "complexity_summary.csv")
    latency = pd.read_csv(source / "latency_summary.csv")
    audit(complexity, latency)

    make_figure(complexity, latency, out)
    make_table(complexity, latency, out)

    single = latency[latency.batch_size.eq(1)][[
        "model", "latency_median_ms_mean", "latency_median_ms_sd"
    ]]
    batch = latency[latency.batch_size.eq(5000)][[
        "model", "throughput_rows_per_second_mean", "throughput_rows_per_second_sd"
    ]]
    source_table = complexity.merge(single, on="model", validate="one_to_one").merge(
        batch, on="model", validate="one_to_one"
    )
    source_table.to_csv(out / "fig6_operational_efficiency_source.csv", index=False)

    output_files = [
        out / "fig6_operational_efficiency.pdf",
        out / "fig6_operational_efficiency.png",
        out / "table_operational_complexity.tex",
        out / "fig6_operational_efficiency_source.csv",
    ]
    result = {
        "completed": True,
        "training_or_inference_performed": False,
        "source_condition": "random_split__original_nine",
        "models": MODELS,
        "training_seeds": [42, 43, 44],
        "source_sha256": {
            "complexity_summary.csv": sha(source / "complexity_summary.csv"),
            "latency_summary.csv": sha(source / "latency_summary.csv"),
            "benchmark_summary.json": sha(source / "summary.json"),
        },
        "output_sha256": {path.name: sha(path) for path in output_files},
        "output_directory": str(out),
    }
    (out / "figure_build_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FIGURE BUILD COMPLETE")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
