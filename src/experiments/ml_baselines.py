"""Traditional ML baselines: MNB, LR, RF, DT, KNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.config import DATASETS, DatasetSpec
from src.data.loader import load_ml_sequences
from src.utils.io import save_json
from src.utils.metrics import classification_metrics


MODELS = {
    "MNB": MultinomialNB(force_alpha=True),
    "LR": LogisticRegression(max_iter=1000, random_state=100),
    "RF": RandomForestClassifier(random_state=100),
    "DT": DecisionTreeClassifier(random_state=100),
    "KNN": KNeighborsClassifier(),
}


def run_ml_baselines(data_dir: Path, output_dir: Path) -> dict:
    all_results: dict = {}
    for spec in tqdm(DATASETS, desc="ML baselines"):
        dataset_results = _run_for_dataset(spec, data_dir)
        all_results[spec.key] = dataset_results
        save_json(
            dataset_results,
            output_dir / "ml_baselines" / f"{spec.key}.json",
        )
    return all_results


def _run_for_dataset(spec: DatasetSpec, data_dir: Path) -> dict:
    x_train, y_train, x_test, y_test, _ = load_ml_sequences(spec, data_dir)
    results = {}
    for name, model in MODELS.items():
        tqdm.write(f"  training {name} on {spec.key}")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
        results[name] = {
            "metrics": classification_metrics(y_test, y_pred, y_prob),
            "predictions": y_pred.tolist(),
        }
    return results
