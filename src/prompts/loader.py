"""Load prompt templates used for LLM classification."""

from pathlib import Path

from src.config import DATASETS, PROMPTS_DIR


def load_prompts() -> dict[str, dict[str, str]]:
    prompts: dict[str, dict[str, str]] = {}
    for spec in DATASETS:
        prompts[spec.key] = {}
        for prompt_type in ("basic", "few_shot"):
            path = PROMPTS_DIR / spec.key / f"{prompt_type}.txt"
            prompts[spec.key][prompt_type] = path.read_text(encoding="utf-8")
    return prompts
