import os.path
import random
import warnings

import numpy as np
from utils.combinations import all_subsets_str
from utils.dataset import build_dataset_by_sample, build_dataset_variable, load_dataset
from utils.plots import plot_correct_predictions_by_class
from utils.search.report import save_search_results
from utils.search.scope import grid_search, jackknife_se
from utils.settings import GetSettings

from scope import SCoPEDistances as SCoPE
from scope.utils.eval_metrics import predictions_to_report
from scope.utils.plot import (
    plot_auc_roc,
    plot_confusion_matrix,
)

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def print_results(report) -> None:
    print("\nResultados finales en test:")
    print(f"  auc_roc (votos)      : {report['auc_roc']}")
    print(f"  distance_auc         : {report.get('distance_auc')}")
    print(f"  f1                   : {report['f1']}")
    print(f"  mcc                  : {report['mcc']}")
    print(f"  balanced_accuracy    : {report['balanced_accuracy']}")

    cm = report["confusion_matrix"]
    print("  confusion_matrix:")
    print("        pred 0  pred 1")
    print(f"true 0    {cm[0][0]:>4}    {cm[0][1]:>4}")
    print(f"true 1    {cm[1][0]:>4}    {cm[1][1]:>4}")

    cm = report["confusion_matrix_normalized"]
    print("  confusion_matrix_normalized:")
    print("        pred 0  pred 1")
    print(f"true 0    {cm[0][0]:>6.3f}  {cm[0][1]:>6.3f}")
    print(f"true 1    {cm[1][0]:>6.3f}  {cm[1][1]:>6.3f}")


def get_dataset(
    settings: GetSettings,
    search_size: list[int] | int,
):
    search, test = load_dataset(
        dataset_name=settings.dataset.dataset_name,
        smiles_column=settings.dataset.smiles_column,
        label_columns=settings.dataset.label_column,
        min_length=settings.dataset.min_length,
        search_size=search_size,
        random_seed=settings.experiment.random_seed,
    )

    return search, test


def _fold_summary(trial, key: str, n_splits: int) -> tuple[float, float]:
    """Media y SE jackknife de `key` sobre los reportes por fold del trial."""
    values = [fr.report[key] for fr in trial.results if fr.report.get(key) is not None]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), jackknife_se(float(np.std(values)), n_splits)


def _search(
    metric: str,
    n_splits: int,
    random_seed: int,
    search_queries: list[str],
    search_y: list[int],
    search_supports: list[dict],
    report_path: str,
) -> SCoPE:
    all_compressors = [
        selection.split(",") for selection in all_subsets_str(["bz2", "zlib", "gzip"])
    ]
    all_dissimilarity_metrics = [
        selection.split(",") for selection in all_subsets_str(["ncd", "cdm", "clm"])
    ]
    param_grid = {
        "compressors": all_compressors,
        "dissimilarity_metrics": all_dissimilarity_metrics,
    }

    results = grid_search(
        queries=search_queries,
        supports=search_supports,
        y_true=search_y,
        param_grid=param_grid,
        metric=metric,
        n_splits=n_splits,
        random_seed=random_seed,
    )

    save_search_results(
        results=results,
        path=report_path,
        metric=metric,
        param_space=param_grid,
    )

    best = results[0]
    se = jackknife_se(best.std_score, n_splits)
    ba, ba_se = _fold_summary(best, "balanced_accuracy", n_splits)

    print(f"\nMejor configuración: {best.params}")
    print(f"  {metric:<20} (search): {best.mean_score:.4f} ± {se:.4f} (SE)")
    if metric != "balanced_accuracy":
        print(f"  {'balanced_accuracy':<20} (search): {ba:.4f} ± {ba_se:.4f} (SE)")

    model = SCoPE(**best.params)

    return model


def _test(
    model: SCoPE,
    test_queries: list[str],
    test_y: list[int],
    test_supports: list[dict],
    report_path: str,
    plot_path: str,
):

    predictions = model.predict(
        kw_samples=test_supports,
        queries=test_queries,
    )

    report = predictions_to_report(
        predictions=predictions,
        y_true=test_y,
        save_path=report_path,
    )

    print_results(report)

    plot_confusion_matrix(
        report["confusion_matrix"],
        labels=["0", "1"],
        save_path=os.path.join(plot_path, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        report["confusion_matrix_normalized"],
        labels=["0", "1"],
        save_path=os.path.join(plot_path, "confusion_matrix_normalized.png"),
        show=True,
        normalize=True,
    )
    plot_correct_predictions_by_class(
        predictions=predictions,
        y_true=test_y,
        n_per_class=5,
        save_dir=plot_path,
    )
    plot_auc_roc(
        report,
        save_path=os.path.join(plot_path, "auc_roc.png"),
        show=False,
    )


def search_scope_variable(
    settings: GetSettings,
    search,
    test,
):
    sample_sizes = settings.experiment.sample_sizes
    random_seed = settings.experiment.random_seed
    metric = settings.optimization.metric
    n_splits = settings.optimization.n_splits

    report_path = os.path.join(settings.optimization_path, "variable")
    plot_path = os.path.join(settings.plots_path, "variable")

    search_queries, search_y, search_supports = build_dataset_variable(
        dataframe=search, sample_sizes=sample_sizes, random_seed=random_seed
    )

    test_queries, test_y, test_supports = build_dataset_variable(
        dataframe=test, sample_sizes=sample_sizes, random_seed=random_seed
    )

    print("\n")
    print("# Samples: Variable")
    print("=" * 120)

    best_model = _search(
        n_splits=n_splits,
        random_seed=random_seed,
        metric=metric,
        search_queries=search_queries,
        search_y=search_y,
        search_supports=search_supports,
        report_path=report_path,
    )

    _test(
        model=best_model,
        test_queries=test_queries,
        test_y=test_y,
        test_supports=test_supports,
        report_path=report_path,
        plot_path=plot_path,
    )

    print("=" * 120)
    print("\n")


def search_scope_sample(
    settings: GetSettings,
    search,
    test,
):
    sample_sizes = settings.experiment.sample_sizes
    random_seed = settings.experiment.random_seed
    metric = settings.optimization.metric
    n_splits = settings.optimization.n_splits

    search_queries_dict, search_y_dict, search_supports_dict = build_dataset_by_sample(
        dataframe=search, sample_sizes=sample_sizes, random_seed=random_seed
    )

    test_queries_dict, test_y_dict, test_supports_dict = build_dataset_by_sample(
        dataframe=test, sample_sizes=sample_sizes, random_seed=random_seed
    )

    for sample_size in sample_sizes:
        print("\n")
        print(f"# Samples: {sample_size}")
        print("=" * 120)

        search_queries, search_y, search_supports = (
            search_queries_dict[sample_size],
            search_y_dict[sample_size],
            search_supports_dict[sample_size],
        )
        test_queries, test_y, test_supports = (
            test_queries_dict[sample_size],
            test_y_dict[sample_size],
            test_supports_dict[sample_size],
        )

        report_path = os.path.join(settings.optimization_path, f"{sample_size}")
        plot_path = os.path.join(settings.plots_path, f"{sample_size}")

        best_model = _search(
            metric=metric,
            search_queries=search_queries,
            search_y=search_y,
            search_supports=search_supports,
            report_path=report_path,
            n_splits=n_splits,
            random_seed=random_seed,
        )

        _test(
            model=best_model,
            test_queries=test_queries,
            test_y=test_y,
            test_supports=test_supports,
            report_path=report_path,
            plot_path=plot_path,
        )
        print("=" * 120)
        print("\n")


if __name__ == "__main__":
    settings_files: list[str] = [
        "settings/clintox.toml",
    ]

    for file_ in settings_files:
        settings = GetSettings(file_)
        random_seed = settings.experiment.random_seed
        sample_sizes = settings.experiment.sample_sizes

        search_data, test_data = get_dataset(
            settings,
            search_size=settings.optimization.test_size,
        )

        search_scope_variable(
            settings=settings,
            search=search_data,
            test=test_data,
        )

        search_scope_sample(
            settings=settings,
            search=search_data,
            test=test_data,
        )
