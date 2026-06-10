"""Aggregate experiment outputs into paper-style result tables."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import DATASETS


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_summary_tables(results_dir: Path) -> dict:
    tables: dict[str, list[dict]] = {}

    for spec in DATASETS:
        rows: list[dict] = []

        ml = _load_json(results_dir / "ml_baselines" / f"{spec.key}.json")
        if ml:
            for model, payload in ml.items():
                metrics = payload["metrics"]
                rows.append(
                    {
                        "model": model,
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "ue_rate": None,
                        "category": "ML",
                    }
                )

        nn = _load_json(results_dir / "nn_baselines" / f"{spec.key}.json")
        if nn:
            for model, payload in nn.items():
                metrics = payload["metrics"]
                rows.append(
                    {
                        "model": model,
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "ue_rate": None,
                        "category": "NN",
                    }
                )

        zsl = _load_json(results_dir / "zero_shot" / f"{spec.key}.json")
        if zsl:
            for model, payload in zsl.items():
                metrics = payload["metrics"]
                rows.append(
                    {
                        "model": model,
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "ue_rate": None,
                        "category": "ZSL",
                    }
                )

        for llm_path in sorted((results_dir / "llm").rglob(f"{spec.key}.json")):
            payload = _load_json(llm_path)
            if not payload:
                continue
            metrics = payload["metrics"]
            prompt = payload.get("prompt_type", "basic")
            backend = payload.get("backend", "llm")
            model = payload.get("model", backend)
            suffix = "(S)" if prompt == "few_shot" else ""
            rows.append(
                {
                    "model": f"{model}{suffix}",
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "ue_rate": metrics.get("ue_rate"),
                    "category": "LLM",
                }
            )

        tables[spec.display_name] = rows

    return tables


def render_markdown(tables: dict) -> str:
    lines = [
        "# Smart Expert System — Experiment Results",
        "",
        "Reproduction of Wang et al. (2024) arXiv:2405.10523.",
        "",
    ]
    for dataset_name, rows in tables.items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append("| Model | ACC (↑) | F1 (↑) | U/E (↓) |")
        lines.append("|-------|---------|--------|---------|")
        for row in rows:
            lines.append(
                f"| {row['model']} | "
                f"{_format_metric(row['accuracy'])} | "
                f"{_format_metric(row['f1_macro'])} | "
                f"{_format_metric(row['ue_rate'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_summary(results_dir: Path) -> Path:
    tables = build_summary_tables(results_dir)
    summary = {
        "paper": "Smart Expert System: Large Language Models as Text Classifiers",
        "arxiv": "https://arxiv.org/abs/2405.10523",
        "tables": tables,
    }
    json_path = results_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    md_path = results_dir / "RESULTS.md"
    md_path.write_text(render_markdown(tables), encoding="utf-8")
    return md_path
