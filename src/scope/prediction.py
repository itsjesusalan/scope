from dataclasses import dataclass

import numpy as np
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GeometryMetrics:
    normalized_distance: float
    fraction_outside: float
    iou: float


@dataclass(frozen=True)
class Prediction:
    scores: dict[int, float]
    predicted_class: int
    dissimilarity_matrix: dict[str, np.ndarray]


@dataclass(frozen=True)
class PolygonPrediction(Prediction):
    convex_hull_clusters: dict[int, BaseGeometry]
    convex_hull_queries: dict[int, BaseGeometry]
    query_points: dict[int, np.ndarray]
    geometry_metrics: dict[int, GeometryMetrics]


@dataclass(frozen=True)
class DistPrediction(Prediction):
    euclidean_distances: dict[int, list[float]]
    cosine_distances: dict[int, list[float]]
    distances: dict[int, list[float]]
    wining_votes: int
    classifier_labels: list[str]
