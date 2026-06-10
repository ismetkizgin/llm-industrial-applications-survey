"""BART and DeBERTa zero-shot classifiers (paper Section V-A)."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm
from transformers import pipeline

from src.config import DATASETS, LABEL_NAMES, DatasetSpec
from src.data.loader import load_split
from src.utils.io import save_json
from src.utils.metrics import classification_metrics


ZERO_SHOT_MODELS = {
    "BART": "facebook/bart-large-mnli",
    "DeBERTa": "microsoft/deberta-large-mnli",
}


def _label_strings(spec: DatasetSpec) -> list[str]:
    return LABEL_NAMES[spec.key]


def _predict_dataset(classifier, spec: DatasetSpec, data_dir: Path) -> dict:
    texts, y_true = load_split(spec, data_dir, "test")
    labels = _label_strings(spec)
    label_to_id = {name: idx for idx, name in enumerate(labels)}

    y_pred = []
    scores = []
    for text in tqdm(texts, desc=f"  {spec.key}", leave=False):
        result = classifier(str(text), labels, multi_label=False)
        y_pred.append(label_to_id[result["labels"][0]])
        scores.append(result["scores"])

    return {
        "metrics": classification_metrics(y_true, y_pred, scores),
        "predictions": y_pred,
    }


def run_zero_shot(data_dir: Path, output_dir: Path) -> dict:
    all_results: dict = {}
    classifiers = {
        name: pipeline("zero-shot-classification", model=model_id)
        for name, model_id in ZERO_SHOT_MODELS.items()
    }
    for spec in tqdm(DATASETS, desc="Zero-shot transformers"):
        dataset_results = {}
        for model_name, classifier in classifiers.items():
            tqdm.write(f"  {model_name} on {spec.key}")
            dataset_results[model_name] = _predict_dataset(
                classifier, spec, data_dir
            )
        all_results[spec.key] = dataset_results
        save_json(dataset_results, output_dir / "zero_shot" / f"{spec.key}.json")
    return all_results
