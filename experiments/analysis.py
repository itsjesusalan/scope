import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.facecolor": "#FCFCFC",
        "figure.facecolor": "#FFFFFF",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    },
)

DATASET_LABELS = {
    "clintox": "ClinTox",
    "bace": "BACE",
    "bbbp": "BBBP",
}


def pretty_dataset(name: str) -> str:
    return DATASET_LABELS.get(str(name).lower(), str(name).upper())


METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "f1_score": "Macro F1",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "MCC",
    "fp_rate": "False positive rate",
    "fn_rate": "False negative rate",
}

METRIC_ORDER = ["auc_roc", "balanced_accuracy", "f1_score", "mcc"]


SAMPLE_ORDER: list[str] = []
SAMPLE_POS: dict[str, float] = {}
SAMPLE_FIXED: list[str] = []
SAMPLE_SPECIAL: list[str] = []
GAP = 1.6


def set_sample_order(*dfs):
    global SAMPLE_ORDER, SAMPLE_POS, SAMPLE_FIXED, SAMPLE_SPECIAL
    vals = set()
    for d in dfs:
        if not d.empty:
            vals |= set(d["samples"].astype(str))
    nums = sorted((v for v in vals if v.isdigit()), key=int)
    others = sorted(v for v in vals if not v.isdigit())

    SAMPLE_FIXED, SAMPLE_SPECIAL = nums, others
    SAMPLE_ORDER = nums + others

    SAMPLE_POS = {s: float(i) for i, s in enumerate(nums)}
    for j, s in enumerate(others):
        SAMPLE_POS[s] = len(nums) - 1 + GAP + j


def _divider_x():
    if not SAMPLE_FIXED or not SAMPLE_SPECIAL:
        return None
    return (SAMPLE_POS[SAMPLE_FIXED[-1]] + SAMPLE_POS[SAMPLE_SPECIAL[0]]) / 2


def set_sample_axis(ax, divider: bool = True):
    ticks = [SAMPLE_POS[s] for s in SAMPLE_ORDER]
    ax.set_xticks(ticks)
    ax.set_xticklabels(SAMPLE_ORDER)
    ax.set_xlim(min(ticks) - 0.5, max(ticks) + 0.5)

    xd = _divider_x()
    if divider and xd is not None:
        ax.axvline(xd, color="0.65", linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)


def split_fixed_special(df):
    is_special = df["samples"].astype(str).isin(SAMPLE_SPECIAL)
    return df[~is_special], df[is_special]


def parse_all_reports_comprehensive(root_dir):
    search_records = []
    eval_records = []

    search_regex = re.compile(r"results/([^/]+)/search_results/(\d+|variable)")
    eval_regex = re.compile(r"results/([^/]+)/search_results/(\d+|variable)")

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".json") and file.startswith("search_report"):
                match = search_regex.search(file_path)
                if not match:
                    continue
                dataset, samples = match.group(1), match.group(2)  # samples siempre str

                try:
                    with open(file_path, "r") as f:
                        report = json.load(f)

                    native_metric = report.get("metric", "balanced_accuracy")

                    for trial in report.get("trials", []):
                        params = trial.get("params", {})
                        compressors = "+".join(sorted(params.get("compressors", [])))
                        metrics = "+".join(
                            sorted(params.get("dissimilarity_metrics", []))
                        )

                        search_records.append(
                            {
                                "dataset": dataset,
                                "samples": samples,
                                "rank": trial.get("rank"),
                                "native_metric": native_metric,
                                "score": trial.get("score"),
                                "std_score": trial.get("std_score"),
                                "compressors": compressors,
                                "metrics": metrics,
                            }
                        )
                except Exception as e:
                    print(f"Error leyendo {file}: {e}")

            elif file.endswith(".json") and file.startswith("evaluation_report"):
                match = eval_regex.search(file_path)
                if not match:
                    continue
                dataset, samples = match.group(1), match.group(2)  # samples siempre str

                try:
                    with open(file_path, "r") as f:
                        report = json.load(f)

                    cm = report.get("confusion_matrix", [[0, 0], [0, 0]])
                    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

                    eval_records.append(
                        {
                            "dataset": dataset,
                            "samples": samples,
                            "auc_roc": report.get("auc_roc"),
                            "f1_score": report.get("f1"),
                            "balanced_accuracy": report.get("balanced_accuracy"),
                            "mcc": report.get("mcc"),
                            "fp_rate": fp / (tn + fp) if (tn + fp) > 0 else 0,
                            "fn_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
                        }
                    )
                except Exception as e:
                    print(f"Error leyendo {file}: {e}")

    return pd.DataFrame(search_records), pd.DataFrame(eval_records)


def plot_learning_curve(
    df_search,
    output_dir: str,
    dataset_name: str,
    figsize: tuple = (10, 5),
):
    plt.figure(figsize=figsize)

    best_trials = df_search[df_search["rank"] == 1].copy()
    best_trials["x"] = best_trials["samples"].map(SAMPLE_POS)
    best_trials = best_trials.sort_values("x")

    base_color = sns.color_palette("flare", 3)[1]

    sns.lineplot(
        data=best_trials,
        x="x",
        y="score",
        marker="o",
        linewidth=3,
        markersize=8,
        color=base_color,
    )

    plt.fill_between(
        best_trials["x"],
        best_trials["score"] - best_trials["std_score"],
        best_trials["score"] + best_trials["std_score"],
        color=base_color,
        alpha=0.15,
    )

    set_sample_axis(plt.gca())
    plt.title(
        f"[{pretty_dataset(dataset_name)}] Parameter Search Validation: Best Score vs Sample Size",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Top Validation Score", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_learning_curve.png"), dpi=300)
    plt.close()


def plot_final_test_metrics(
    df_eval,
    output_dir: str,
    dataset_name: str,
    figsize: tuple = (7.2, 4.6),
    layout: str = "single",
    ylim: tuple | None = (-0.06, 1.0),
    chance_line: bool = False,
    dodge: float = 0.09,
):
    if df_eval.empty:
        return

    value_vars = [m for m in METRIC_ORDER if m in df_eval.columns]
    df_long = df_eval.melt(
        id_vars=["samples"],
        value_vars=value_vars,
        var_name="metric",
        value_name="Value",
    )
    df_long["x"] = df_long["samples"].astype(str).map(SAMPLE_POS)
    df_long["Metric"] = df_long["metric"].map(METRIC_LABELS)
    df_long = df_long.sort_values("x")

    labels = [METRIC_LABELS[m] for m in value_vars]
    palette = dict(zip(labels, sns.color_palette("flare", len(labels))))
    markers = dict(zip(labels, ["o", "s", "^", "D"][: len(labels)]))
    linestyles = dict(
        zip(labels, ["-", (0, (5, 1.5)), (0, (1, 1.2)), (0, (4, 1.2, 1, 1.2))])
    )

    offsets = {
        label: (i - (len(labels) - 1) / 2) * dodge for i, label in enumerate(labels)
    }

    title = f"{pretty_dataset(dataset_name)} — test performance vs support-set size"

    if layout == "single":
        fig, ax = plt.subplots(figsize=figsize)

        fixed, special = split_fixed_special(df_long)

        for label in labels:
            dx = offsets[label]
            sub = fixed[fixed["Metric"] == label]
            ax.plot(
                sub["x"] + dx,
                sub["Value"],
                marker=markers[label],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=1.8,
                linestyle=linestyles[label],
                color=palette[label],
                label=label,
                zorder=3,
            )
            sp = special[special["Metric"] == label]
            ax.plot(
                sp["x"] + dx,
                sp["Value"],
                marker=markers[label],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                color=palette[label],
                zorder=3,
            )

        if chance_line:
            ax.axhline(
                0.5,
                color="0.45",
                linewidth=1.0,
                linestyle=":",
                zorder=0,
                label="Chance (AUC-ROC, bal. acc.)",
            )

        set_sample_axis(ax)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Support-set size $k$", fontsize=11)
        ax.set_ylabel("Test score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        handles, lab = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            lab,
            fontsize=9,
            ncol=min(len(lab), 4),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
        )

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "final_test_evaluation.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        return

    ncol = 2
    nrow = -(-len(labels) // ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )

    for ax, label in zip(axes.flat, labels):
        sub = df_long[df_long["Metric"] == label].sort_values("x")
        fixed, special = split_fixed_special(sub)
        ax.plot(
            fixed["x"],
            fixed["Value"],
            marker="o",
            markersize=5,
            linewidth=2.0,
            color=palette[label],
            zorder=3,
        )
        ax.plot(
            special["x"],
            special["Value"],
            marker="o",
            markersize=6,
            linestyle="none",
            color=palette[label],
            zorder=3,
        )
        set_sample_axis(ax)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=11)

    for ax in list(axes.flat)[len(labels) :]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Support-set size $k$", fontsize=11)
    for row in axes:
        row[0].set_ylabel("Test score", fontsize=11)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "final_test_evaluation.png"), dpi=300)
    plt.close(fig)


def plot_validation_vs_test(
    df_search,
    df_eval,
    output_dir: str,
    dataset_name: str,
    target_metric: str,
    figsize: tuple = (10, 5),
):
    if df_search.empty or df_eval.empty:
        return

    test_metric_col = (
        "balanced_accuracy" if target_metric == "balanced_accuracy" else target_metric
    )
    if test_metric_col not in df_eval.columns:
        return

    val_best = df_search[df_search["rank"] == 1][["samples", "score"]].rename(
        columns={"score": "Parameter Search Validation"}
    )
    test_results = df_eval[["samples", test_metric_col]].rename(
        columns={test_metric_col: "Final Test Evaluation"}
    )

    merged = pd.merge(val_best, test_results, on=["samples"])
    if merged.empty:
        return

    merged_long = merged.melt(
        id_vars=["samples"],
        value_vars=["Parameter Search Validation", "Final Test Evaluation"],
        var_name="Phase",
        value_name="Value",
    )
    merged_long["x"] = merged_long["samples"].map(SAMPLE_POS)

    plt.figure(figsize=figsize)
    sns.lineplot(
        data=merged_long,
        x="x",
        y="Value",
        style="Phase",
        hue="Phase",
        markers=True,
        linewidth=2.5,
        palette="flare",
    )

    set_sample_axis(plt.gca())
    plt.title(
        f"[{pretty_dataset(dataset_name)}] Generalization Gap: Parameter Search vs Final Test ({target_metric})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel(f"Score ({target_metric})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "search_vs_test_gap.png"), dpi=300)
    plt.close()


def plot_error_rates_evolution(
    df_eval, output_dir: str, dataset_name: str, figsize: tuple = (9, 5.5)
):
    if df_eval.empty:
        return

    df_errors = df_eval.melt(
        id_vars=["samples"],
        value_vars=["fp_rate", "fn_rate"],
        var_name="Error_Type",
        value_name="Rate",
    )
    df_errors["x"] = df_errors["samples"].map(SAMPLE_POS)

    plt.figure(figsize=figsize)
    sns.lineplot(
        data=df_errors,
        x="x",
        y="Rate",
        hue="Error_Type",
        style="Error_Type",
        markers=True,
        linewidth=2.5,
        palette="flare",
    )

    set_sample_axis(plt.gca())
    plt.title(
        f"[{pretty_dataset(dataset_name)}] Error Analysis: False Positive vs False Negative Rates (Final Test Evaluation)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Error Rate", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_error_rates.png"), dpi=300)
    plt.close()


def plot_stability_scatter(
    df_search, output_dir: str, dataset_name: str, figsize: tuple = (9, 5.5)
):
    viable_df = df_search

    local_order = [s for s in SAMPLE_ORDER if s in set(viable_df["samples"])]

    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=viable_df,
        x="score",
        y="std_score",
        hue="samples",
        hue_order=local_order,
        size="samples",
        size_order=local_order,
        palette=sns.color_palette("flare", len(local_order)),
        alpha=0.7,
        sizes=(40, 200),
    )

    plt.title(
        f"[{pretty_dataset(dataset_name)}] Optimizer Stability: Validation Score vs Standard Deviation",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Parameter Search Validation Score", fontsize=11)
    plt.ylabel("Standard Deviation (std_score)", fontsize=11)
    plt.legend(title="Sample Size", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimizer_stability_scatter.png"), dpi=300)
    plt.close()


def plot_compressor_metric_heatmap(
    df_search, output_dir: str, dataset_name: str, figsize: tuple = (10, 6.5)
):
    viable_df = df_search
    if viable_df.empty:
        return

    pivot_table = viable_df.pivot_table(
        values="score", index="compressors", columns="metrics", aggfunc="mean"
    )

    dynamic_width = max(figsize[0], pivot_table.shape[1] * 1.3 + 2)
    dynamic_height = max(figsize[1], pivot_table.shape[0] * 0.8 + 2)

    plt.figure(figsize=(dynamic_width, dynamic_height))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="flare",
        linewidths=0.5,
        square=False,
        cbar_kws={"label": "Mean Validation Score", "shrink": 0.8},
    )

    plt.title(
        f"[{pretty_dataset(dataset_name)}] Synergy Matrix: Compressor vs Metric Interaction (Parameter Search)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.xlabel("Dissimilarity Metrics", fontsize=11, labelpad=10)
    plt.ylabel("Compressor Pipelines", fontsize=11, labelpad=10)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "compressor_metric_synergy_heatmap.png"), dpi=300
    )
    plt.close()


def plot_global_top_parameters_census(
    df_search, output_dir: str, figsize: tuple = (10, 6)
):
    best_configs = df_search[df_search["rank"] == 1].copy()
    if best_configs.empty:
        return

    best_configs["combo"] = (
        best_configs["compressors"] + "\n(" + best_configs["metrics"] + ")"
    )
    counts = best_configs["combo"].value_counts().reset_index()
    counts.columns = ["Configuration", "Count"]

    calculated_height = max(figsize[1], len(counts) * 0.7 + 2)
    plt.figure(figsize=(figsize[0], calculated_height))

    sns.barplot(
        data=counts,
        x="Count",
        y="Configuration",
        hue="Configuration",
        palette="flare_r",
        legend=False,
        edgecolor=".2",
        width=0.8,
    )

    plt.title(
        "Global Parameter Census: Most Frequent Rank-1 Configs (Search Parameters)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Number of Wins (Across all Datasets & Samples)", fontsize=11)
    plt.ylabel("Configuration (Compressor + Metric)", fontsize=11)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_top_parameters_census.png"), dpi=300)
    plt.close()


def plot_best_param_matrix(
    df_search, output_dir: str, target_metric: str, figsize: tuple = (14, 6)
):
    best_configs = df_search[df_search["rank"] == 1].copy()
    if best_configs.empty:
        return

    best_configs["label"] = (
        best_configs["compressors"]
        + "\n"
        + best_configs["metrics"]
        + "\n"
        + best_configs["score"].map(lambda x: f"({x:.3f})")
    )

    label_pivot = best_configs.pivot(index="dataset", columns="samples", values="label")
    score_pivot = best_configs.pivot(index="dataset", columns="samples", values="score")

    cols = [s for s in SAMPLE_ORDER if s in score_pivot.columns]
    label_pivot = label_pivot.reindex(columns=cols)
    score_pivot = score_pivot.reindex(columns=cols)

    matrix_df = label_pivot.fillna("N/A")
    score_pivot = score_pivot.fillna(0)

    calculated_height = max(figsize[1], len(matrix_df.index) * 1.8 + 2)
    plt.figure(figsize=(figsize[0], calculated_height))

    sns.heatmap(
        score_pivot,
        annot=matrix_df.values,
        fmt="",
        cmap="flare",
        linewidths=0.8,
        cbar_kws={"label": f"Search Validation Score ({target_metric})"},
        annot_kws={"fontsize": 9, "fontweight": "medium"},
    )

    plt.title(
        f"Optimal Parameter Landscape: Rank 1 Combinations ({target_metric})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Number of Samples", fontsize=11)
    plt.ylabel("Datasets", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "optimal_parameter_landscape_matrix.png"), dpi=300
    )
    plt.close()


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    TARGET_METRIC = "balanced_accuracy"

    ROOT_DATA_DIR = "./results"
    BASE_OUTPUT_DIR = os.path.join(ROOT_DATA_DIR, "plots_analisis")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    df_search_all, df_eval_all = parse_all_reports_comprehensive(ROOT_DATA_DIR)
    set_sample_order(df_search_all, df_eval_all)

    datasets_search = (
        df_search_all["dataset"].unique() if not df_search_all.empty else []
    )
    datasets_eval = df_eval_all["dataset"].unique() if not df_eval_all.empty else []
    all_datasets = sorted(set(datasets_search) | set(datasets_eval))

    if not all_datasets:
        print("No se encontraron registros validos mapeados en el directorio raiz.")
    else:
        if not df_search_all.empty:
            plot_global_top_parameters_census(df_search_all, BASE_OUTPUT_DIR)
            plot_best_param_matrix(df_search_all, BASE_OUTPUT_DIR, TARGET_METRIC)

        for current_dataset in all_datasets:
            dataset_plot_dir = os.path.join(BASE_OUTPUT_DIR, current_dataset)
            os.makedirs(dataset_plot_dir, exist_ok=True)

            df_search = (
                df_search_all[df_search_all["dataset"] == current_dataset]
                if not df_search_all.empty
                else pd.DataFrame()
            )
            df_eval = (
                df_eval_all[df_eval_all["dataset"] == current_dataset]
                if not df_eval_all.empty
                else pd.DataFrame()
            )

            if not df_search.empty:
                plot_learning_curve(df_search, dataset_plot_dir, current_dataset)
                plot_stability_scatter(df_search, dataset_plot_dir, current_dataset)
                plot_compressor_metric_heatmap(
                    df_search, dataset_plot_dir, current_dataset
                )

            if not df_eval.empty:
                plot_final_test_metrics(df_eval, dataset_plot_dir, current_dataset)
                plot_error_rates_evolution(df_eval, dataset_plot_dir, current_dataset)

            if not df_search.empty and not df_eval.empty:
                plot_validation_vs_test(
                    df_search, df_eval, dataset_plot_dir, current_dataset, TARGET_METRIC
                )
