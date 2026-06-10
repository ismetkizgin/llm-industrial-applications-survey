"""Text preprocessing for ML/NN baselines (paper Section V-A)."""

import re

import nltk
from nltk.corpus import stopwords


def ensure_nltk_data() -> None:
    for resource in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.data.find(
                "corpora/stopwords"
                if resource == "stopwords"
                else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


def text_cleaner(text: str) -> str:
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)
