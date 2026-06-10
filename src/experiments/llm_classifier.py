"""LLM zero-shot / few-shot classification with U/E rate metric."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

from src.config import DATASETS, LABEL_NAMES
from src.utils.env import get_env, load_env

load_env()
from src.data.loader import load_split
from src.prompts.loader import load_prompts
from src.utils.io import save_json
from src.utils.metrics import classification_metrics, ue_rate

REFUSAL_PATTERNS = [
    r"cannot classify",
    r"can't classify",
    r"unable to",
    r"i'm sorry",
    r"i am sorry",
    r"not able to",
    r"refuse",
    r"inappropriate",
]


def _parse_llm_response(
    response: str,
    valid_labels: list[str],
) -> tuple[int | None, bool, bool]:
    """Return (label_id, is_uncertain, is_error)."""
    text = response.strip().lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text):
            return None, True, False

    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group())
            value = str(payload.get("value", "")).strip().lower()
            for idx, label in enumerate(valid_labels):
                if value == label.lower():
                    return idx, False, False
        except json.JSONDecodeError:
            pass

    lowered = response.lower()
    for idx, label in enumerate(valid_labels):
        if label.lower() in lowered:
            return idx, False, False

    return None, False, True


def _build_prompt(template: str, text: str) -> str:
    return f"{template}{text}"


def _classify_with_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or export it in your shell."
        )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
    )
    return response.choices[0].message.content or ""


def _classify_with_local_hf(prompt: str, model_id: str) -> str:
    from transformers import pipeline

    if not hasattr(_classify_with_local_hf, "_pipe"):
        _classify_with_local_hf._pipe = pipeline(
            "text-generation",
            model=model_id,
            max_new_tokens=64,
            do_sample=False,
        )
    outputs = _classify_with_local_hf._pipe(prompt)
    generated = outputs[0]["generated_text"]
    if generated.startswith(prompt):
        return generated[len(prompt) :].strip()
    return generated.strip()


def run_llm_experiments(
    data_dir: Path,
    output_dir: Path,
    prompt_type: str = "basic",
    backend: str = "auto",
    model_name: str = "gpt-3.5-turbo",
    local_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_samples: int | None = None,
) -> dict:
    prompts = load_prompts()
    all_results: dict = {}

    if backend == "auto":
        backend = "openai" if get_env("OPENAI_API_KEY") else "local"

    for spec in tqdm(DATASETS, desc="LLM classification"):
        texts, y_true = load_split(spec, data_dir, "test")
        if max_samples is not None:
            texts = texts[:max_samples]
            y_true = y_true[:max_samples]

        labels = LABEL_NAMES[spec.key]
        template = prompts[spec.key][prompt_type]
        y_pred: list[int] = []
        uncertain = 0
        errors = 0
        raw_responses: list[dict] = []

        for text in tqdm(texts, desc=f"  {spec.key}", leave=False):
            prompt = _build_prompt(template, str(text))
            if backend == "openai":
                response = _classify_with_openai(prompt, model_name)
            else:
                response = _classify_with_local_hf(prompt, local_model)

            label_id, is_uncertain, is_error = _parse_llm_response(response, labels)
            if is_uncertain:
                uncertain += 1
                y_pred.append(-1)
            elif is_error or label_id is None:
                errors += 1
                y_pred.append(-1)
            else:
                y_pred.append(label_id)

            raw_responses.append(
                {
                    "response": response,
                    "parsed_label": label_id,
                    "uncertain": is_uncertain,
                    "error": is_error,
                }
            )

        valid_mask = [p >= 0 for p in y_pred]
        metrics = classification_metrics(
            y_true[valid_mask],
            [p for p in y_pred if p >= 0],
        )
        metrics["ue_rate"] = ue_rate(len(texts), uncertain, errors)
        metrics["uncertain_count"] = uncertain
        metrics["error_count"] = errors

        result = {
            "model": model_name if backend == "openai" else local_model,
            "backend": backend,
            "prompt_type": prompt_type,
            "metrics": metrics,
            "predictions": y_pred,
            "raw_responses": raw_responses[:20],
        }
        all_results[spec.key] = result
        save_json(
            result,
            output_dir / "llm" / backend / prompt_type / f"{spec.key}.json",
        )

    return all_results
