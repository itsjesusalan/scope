import os
import tomllib
from typing import Any, Dict, List, Union


class ExperimentSettings:
    def __init__(self, cfg):
        self.study_name: str = cfg["study_name"]
        self.sample_sizes: List[int] = cfg["sample_sizes"]
        self.random_seed: int = cfg["random_seed"]


class OptimizationSettings:
    def __init__(self, cfg):
        self.metric: str = cfg["metric"]
        self.test_size: float = cfg["test_size"]
        self.n_splits: int = cfg["n_splits"]


class DatasetSettings:
    def __init__(self, cfg):
        self.dataset_name: str = cfg["dataset_name"]
        self.smiles_column: str = cfg["smiles_column"]
        self.label_column: Union[List[str], str] = cfg["label_column"]
        self.min_length: int = cfg["min_length"]


class GetSettings:
    def __init__(self, settings_file: str):
        with open(settings_file, "rb") as f:
            config = tomllib.load(f)

        self.experiment = ExperimentSettings(config["experiment"])
        self.dataset = DatasetSettings(config["dataset"])
        self.optimization = OptimizationSettings(config["optimization"])

        results_path = os.path.join(
            config["paths"]["results_path"],
            self.experiment.study_name,
        )
        self.results_path = results_path
        self.plots_path = os.path.join(results_path, "plots")
        self.optimization_path = os.path.join(results_path, "search_results")

        for path in [results_path, self.plots_path, self.optimization_path]:
            os.makedirs(path, exist_ok=True)
