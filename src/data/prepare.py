"""Prepare stratified mini datasets (10k train / 800 test per paper)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASETS,
    ENCODING,
    MINI_TEST_SIZE,
    MINI_TRAIN_SIZE,
    RANDOM_STATE,
    DatasetSpec,
)


def _validate_dataset(data: pd.DataFrame, allowed_values: set) -> pd.DataFrame:
    mask = (
        data["label"].isin(allowed_values)
        & data["label"].notna()
        & data["text"].notna()
        & data["text"].apply(lambda x: isinstance(x, str))
        & (data["text"].str.len() > 0)
        & (data["label"].astype(str).str.len() > 0)
    )
    return data.loc[mask].copy()


def _load_raw_csv(path: Path, spec: DatasetSpec) -> pd.DataFrame:
    data = pd.read_csv(path, encoding=ENCODING)
    if isinstance(spec.column_renames, dict):
        data = data.rename(columns=spec.column_renames)
    elif isinstance(spec.column_renames, list):
        data.columns = spec.column_renames
    data = _validate_dataset(data, set(spec.label_map.keys()))
    data["label"] = data["label"].map(spec.label_map).astype(np.int64)
    return data[["text", "label"]]


def _create_mini(data: pd.DataFrame, count: int, dest: Path) -> None:
    if len(data) <= count:
        data.to_csv(dest, index=False)
        return
    _, mini = train_test_split(
        data,
        test_size=count / len(data),
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    mini.to_csv(dest, index=False)


def prepare_dataset(spec: DatasetSpec, data_dir: Path) -> None:
    train_mini = data_dir / f"{spec.key}_train_mini.csv"
    test_mini = data_dir / f"{spec.key}_test_mini.csv"
    if train_mini.exists() and test_mini.exists():
        return

    if spec.key == "Corona_NLP_new":
        train = _load_raw_csv(data_dir / "Corona_NLP_train.csv", spec)
        test = _load_raw_csv(data_dir / "Corona_NLP_test.csv", spec)
        train.to_csv(data_dir / f"{spec.key}_train.csv", index=False)
        test.to_csv(data_dir / f"{spec.key}_test.csv", index=False)
    else:
        raw = _load_raw_csv(data_dir / spec.original_files[0], spec)
        train, test = train_test_split(
            raw,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=raw["label"],
        )
        train.to_csv(data_dir / f"{spec.key}_train.csv", index=False)
        test.to_csv(data_dir / f"{spec.key}_test.csv", index=False)

    _create_mini(train, MINI_TRAIN_SIZE, train_mini)
    _create_mini(test, MINI_TEST_SIZE, test_mini)


def prepare_all(data_dir: Path) -> None:
    for spec in DATASETS:
        prepare_dataset(spec, data_dir)
