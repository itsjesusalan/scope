from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from scope import SCoPEDistances as SCoPE
from scope.utils.eval_metrics import Report, predictions_to_report


@dataclass
class FoldResult:
    fold: int
    score: float
    report: Report


@dataclass
class TrialResult:
    params: dict[str, Any]
    mean_score: float
    std_score: float
    results: list[FoldResult] = field(default_factory=list)


def jackknife_se(std: float, n_splits: int) -> float:
    """Delete-a-group jackknife SE a partir de np.std (ddof=0)."""
    return float(np.sqrt(n_splits - 1) * std)


def _scope_run_(
    fold_idx: int,
    eval_idx: np.ndarray,
    predictions: list,
    y_true: np.ndarray,
    metric: str,
) -> FoldResult:

    report = predictions_to_report(
        predictions=[predictions[i] for i in eval_idx],
        y_true=y_true[eval_idx].tolist(),
    )

    return FoldResult(fold=fold_idx, score=report[metric], report=report)


def grid_search(
    queries: list[str],
    supports: list[dict],
    y_true: list[int],
    param_grid: dict[str, list],
    metric: str = "balanced_accuracy",
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[TrialResult]:
    y_true = np.array(y_true)
    indices = np.arange(len(queries))

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    results: list[TrialResult] = []

    print(f"Grid search: {metric} over {len(combos)} combinations of {keys}")

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    splits = list(kfold.split(indices, y_true))

    pbar = tqdm(combos, desc="Grid search", unit="trial")

    for combo in pbar:
        params = dict(zip(keys, combo))
        model = SCoPE(**params)

        predictions = model.predict(queries=queries, kw_samples=supports)

        fold_result = [
            _scope_run_(
                fold_idx=k,
                eval_idx=eval_idx,
                predictions=predictions,
                y_true=y_true,
                metric=metric,
            )
            for k, (eval_idx, _) in enumerate(splits)
        ]

        scores = [fr.score for fr in fold_result]

        trial = TrialResult(
            params=params,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            results=fold_result,
        )

        results.append(trial)
        se = jackknife_se(trial.std_score, n_splits)
        pbar.set_postfix({metric: f"{trial.mean_score:.4f} ± {se:.4f} (SE)"})

    results.sort(
        key=lambda r: r.mean_score - jackknife_se(r.std_score, n_splits),
        reverse=True,
    )

    return results
