import json
from pathlib import Path


def consolidate_experiments(root_dir: str = ".") -> None:
    root = Path(root_dir)
    records = []

    # Iterate over dataset directories (bace, bbbp, clintox, etc.)
    for dataset_dir in [d for d in root.iterdir() if d.is_dir()]:
        print(dataset_dir)
        dataset_name = dataset_dir.name

        # Locate search_results/scope folder
        scope_path = dataset_dir / "search_results"
        if not scope_path.exists():
            continue

        # Iterate over sample setting subdirectories (3, 5, 8, variable, etc.)
        for sample_dir in [s for s in scope_path.iterdir() if s.is_dir()]:
            sample_setting = sample_dir.name

            eval_file = next(sample_dir.glob("evaluation_report_*.json"), None)
            search_file = next(sample_dir.glob("search_report_*.json"), None)

            eval_data = json.loads(eval_file.read_text()) if eval_file else None
            search_data = json.loads(search_file.read_text()) if search_file else None

            records.append(
                {
                    "dataset": dataset_name,
                    "sample_setting": sample_setting,
                    "search_report": search_data,
                    "evaluation_report": eval_data,
                }
            )

    output_path = root / "consolidated_experiments.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Successfully consolidated {len(records)} experiments into {output_path}")


if __name__ == "__main__":
    consolidate_experiments(root_dir="results/")
