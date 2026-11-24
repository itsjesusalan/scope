import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .compression import DissimilarityMatrix
from .prediction import Prediction


class _SCoPE(ABC):
    def __init__(
        self,
        compressors: list[str],
        join_string: str = " ",
        keep_similar: bool = False,
        class_weights: dict[int, float] | None = None,
        dissimilarity_metric_names: list[str] | None = None,
    ):
        if dissimilarity_metric_names is None:
            dissimilarity_metric_names = ["ncd", "cdm"]

        self._dissimilarity_matrix = DissimilarityMatrix(
            compressor_names=compressors,
            dissimilarity_metric_names=dissimilarity_metric_names,
            join_string=join_string,
            keep_similar=keep_similar,
        )
        self._class_weights = class_weights

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(
        self,
        kw_samples: list[dict[int | str, list[str]]],
        queries: list[str],
    ) -> list[Prediction]:
        dissimilarity_matrices = self._dissimilarity_matrix(
            queries=queries,
            supports=kw_samples,
        )

        predictions = [
            self._predict(dissimilarity_matrix=dissimilarity_matrix)
            for dissimilarity_matrix in dissimilarity_matrices
        ]

        return predictions

    @abstractmethod
    def _predict(
        self,
        dissimilarity_matrix: dict[str, np.ndarray],
    ) -> Any: ...

    @abstractmethod
    def _get_predicted_class(
        self,
        scores: dict[int, float],
    ) -> int: ...
