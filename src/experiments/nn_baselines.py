"""RNN, LSTM, GRU neural network baselines."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tqdm import tqdm

from src.config import DATASETS, DatasetSpec
from src.data.loader import load_ml_sequences
from src.utils.io import save_json
from src.utils.metrics import classification_metrics


def _setup_device() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _build_model(layer_cls, units, input_length, num_classes, vocab_size):
    _setup_device()
    embedding_dim = 16
    return tf.keras.Sequential(
        [
            layers.Embedding(vocab_size, embedding_dim, input_length=input_length),
            layer_cls(units, return_sequences=True),
            layers.GlobalMaxPooling1D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def _train_one(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    name: str,
    epochs: int,
    batch_size: int,
) -> dict:
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=1)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        verbose=0,
        validation_split=0.15,
        batch_size=batch_size,
        callbacks=[callback],
    )
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(np.round(y_prob), axis=1)
    return {
        "metrics": classification_metrics(y_test, y_pred, y_prob),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "predictions": y_pred.tolist(),
    }


def run_nn_baselines(data_dir: Path, output_dir: Path) -> dict:
    all_results: dict = {}
    configs = [
        ("RNN", layers.SimpleRNN, 20, 5),
        ("LSTM", layers.LSTM, 15, 8),
        ("GRU", layers.GRU, 15, 6),
    ]
    for spec in tqdm(DATASETS, desc="NN baselines"):
        x_train, y_train, x_test, y_test, vocab_size = load_ml_sequences(
            spec, data_dir
        )
        dataset_results = {}
        for model_name, layer_cls, units, epochs in configs:
            tqdm.write(f"  training {model_name} on {spec.key}")
            model = _build_model(
                layer_cls,
                units,
                x_train.shape[1],
                spec.num_classes,
                vocab_size + 1,
            )
            dataset_results[model_name] = _train_one(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                model_name,
                epochs,
                32,
            )
        all_results[spec.key] = dataset_results
        save_json(dataset_results, output_dir / "nn_baselines" / f"{spec.key}.json")
    return all_results
