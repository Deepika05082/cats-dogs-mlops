import argparse
import json
import random
from pathlib import Path

import httpx


LABELS = {"Cat": "cat", "Dog": "dog"}


def collect_samples(data_path: Path, samples_per_class: int, seed: int):
    random_generator = random.Random(seed)
    samples = []
    for folder_name, label in LABELS.items():
        files = [path for path in (data_path / folder_name).iterdir() if path.is_file()]
        if len(files) < samples_per_class:
            raise ValueError(f"Not enough images in {data_path / folder_name}")
        samples.extend((path, label) for path in random_generator.sample(files, samples_per_class))
    return samples


def calculate_metrics(results):
    confusion = {"cat": {"cat": 0, "dog": 0}, "dog": {"cat": 0, "dog": 0}}
    for result in results:
        confusion[result["true_label"]][result["predicted_label"]] += 1

    total = len(results)
    correct = sum(result["true_label"] == result["predicted_label"] for result in results)
    precision_values = []
    recall_values = []
    for label in LABELS.values():
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in LABELS.values() if other != label)
        false_negative = sum(confusion[label][other] for other in LABELS.values() if other != label)
        precision_values.append(true_positive / (true_positive + false_positive) if true_positive + false_positive else 0)
        recall_values.append(true_positive / (true_positive + false_negative) if true_positive + false_negative else 0)

    return {
        "sample_count": total,
        "accuracy": correct / total if total else 0,
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "confusion_matrix": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a deployed cats and dogs API.")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--data-path", type=Path, default=Path("data/PetImages"))
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/post_deployment_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = []
    with httpx.Client(timeout=60) as client:
        for image_path, true_label in collect_samples(args.data_path, args.samples_per_class, args.seed):
            with image_path.open("rb") as image_file:
                response = client.post(args.url, files={"file": (image_path.name, image_file, "image/jpeg")})
            response.raise_for_status()
            predicted_label = response.json()["prediction"]
            results.append({"true_label": true_label, "predicted_label": predicted_label})

    report = calculate_metrics(results)
    report["endpoint"] = args.url
    report["results"] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()