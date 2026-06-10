"""Configuration for Smart Expert System text classification experiments."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

MINI_TRAIN_SIZE = 10_000
MINI_TEST_SIZE = 800
RANDOM_STATE = 42
ENCODING = "ISO-8859-1"

GITHUB_RAW = (
    "https://raw.githubusercontent.com/yeyimilk/llm-zero-shot-classifiers/main/dataset"
)

SMS_LABELS = {"normal": 0, "spam": 1}
CORONA_LABELS = {
    "Extremely Negative": 0,
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "Extremely Positive": 2,
}
ECOMMERCE_LABELS = {
    "Household": 0,
    "Books": 1,
    "Clothing & Accessories": 2,
    "Electronics": 3,
}
FINANCIAL_LABELS = {"positive": 2, "neutral": 1, "negative": 0}

LABEL_NAMES = {
    "Corona_NLP_new": ["negative", "neutral", "positive"],
    "ecommerceDataset": [
        "Household",
        "Books",
        "Clothing & Accessories",
        "Electronics",
    ],
    "sms_spam": ["normal", "spam"],
    "financial_sentiment": ["negative", "neutral", "positive"],
}


@dataclass
class DatasetSpec:
    key: str
    display_name: str
    num_classes: int
    label_map: dict
    original_files: list[str] = field(default_factory=list)
    column_renames: dict | list | None = None


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        key="Corona_NLP_new",
        display_name="COVID-19 Tweets",
        num_classes=3,
        label_map=CORONA_LABELS,
        original_files=["Corona_NLP_train.csv", "Corona_NLP_test.csv"],
        column_renames={"Sentiment": "label", "OriginalTweet": "text"},
    ),
    DatasetSpec(
        key="ecommerceDataset",
        display_name="E-Commerce Products",
        num_classes=4,
        label_map=ECOMMERCE_LABELS,
        original_files=["ecommerceDataset.csv"],
        column_renames=["label", "text"],
    ),
    DatasetSpec(
        key="sms_spam",
        display_name="SMS Spam",
        num_classes=2,
        label_map=SMS_LABELS,
        original_files=["sms_spam.csv"],
        column_renames={"label": "label", "text": "text"},
    ),
    DatasetSpec(
        key="financial_sentiment",
        display_name="Economic Texts",
        num_classes=3,
        label_map=FINANCIAL_LABELS,
        original_files=["financial_sentiment.csv"],
        column_renames=["label", "text"],
    ),
]
