"""Dataset loading for experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.config import DATASETS, ENCODING, DatasetSpec
from src.utils.text_clean import text_cleaner


def load_split(
    spec: DatasetSpec,
    data_dir: Path,
    split: str = "test",
    mini: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    suffix = "_mini" if mini else ""
    path = data_dir / f"{spec.key}_{split}{suffix}.csv"
    data = pd.read_csv(path, encoding=ENCODING)
    return data["text"].to_numpy(), data["label"].to_numpy()


def load_ml_sequences(
    spec: DatasetSpec,
    data_dir: Path,
    num_words: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    x_train, y_train = load_split(spec, data_dir, "train")
    x_test, y_test = load_split(spec, data_dir, "test")

    x_train = pd.Series(x_train).apply(text_cleaner)
    x_test = pd.Series(x_test).apply(text_cleaner)

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)
    x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), padding="post")
    x_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(x_test),
        padding="post",
        maxlen=x_train_seq.shape[1],
    )
    vocab_size = len(tokenizer.word_index) if num_words is None else num_words
    return x_train_seq, y_train, x_test_seq, y_test, vocab_size
