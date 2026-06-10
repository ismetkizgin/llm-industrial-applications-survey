#!/usr/bin/env python3
"""End-to-end experiment pipeline for arXiv:2405.10523."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import get_env, get_env_int, load_env

load_env()

from src.config import DATA_DIR, DATASETS, RESULTS_DIR
from src.data.download import download_datasets
from src.data.prepare import prepare_all
from src.experiments.llm_classifier import run_llm_experiments
from src.experiments.ml_baselines import run_ml_baselines
from src.experiments.nn_baselines import run_nn_baselines
from src.experiments.zero_shot import run_zero_shot
from src.summarize import write_summary
from src.utils.io import ensure_dir, save_json, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Smart Expert System text classification experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for all experiment outputs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing prepared CSV datasets.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets from the reference repository.",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip traditional ML baselines.",
    )
    parser.add_argument(
        "--skip-nn",
        action="store_true",
        help="Skip RNN/LSTM/GRU baselines.",
    )
    parser.add_argument(
        "--skip-zero-shot",
        action="store_true",
        help="Skip BART/DeBERTa zero-shot models.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM prompting experiments.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["auto", "openai", "local"],
        default=get_env("LLM_BACKEND", "auto"),
        help="LLM inference backend.",
    )
    parser.add_argument(
        "--llm-model",
        default=get_env("OPENAI_MODEL", "gpt-3.5-turbo"),
        help="OpenAI model name when using openai backend.",
    )
    parser.add_argument(
        "--llm-local-model",
        default=get_env("LLM_LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="HuggingFace model for local LLM backend.",
    )
    parser.add_argument(
        "--llm-prompt",
        choices=["basic", "few_shot", "both"],
        default=get_env("LLM_PROMPT", "basic"),
        help="Prompt strategy for LLM experiments.",
    )
    parser.add_argument(
        "--llm-max-samples",
        type=int,
        default=get_env_int("LLM_MAX_SAMPLES"),
        help="Limit LLM test samples per dataset (useful for quick runs).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip heavy zero-shot models and limit LLM to 50 samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("pipeline")

    output_dir = args.output_dir or (RESULTS_DIR / f"run_{timestamp()}")
    ensure_dir(output_dir)
    ensure_dir(args.data_dir)

    run_config = {
        "paper": "arXiv:2405.10523",
        "datasets": [spec.key for spec in DATASETS],
        "args": vars(args),
    }
    save_json(run_config, output_dir / "run_config.json")

    if not args.skip_download:
        log.info("Downloading datasets...")
        download_datasets(args.data_dir)

    log.info("Preparing stratified mini datasets...")
    prepare_all(args.data_dir)

    if args.fast:
        args.skip_zero_shot = True
        if args.llm_max_samples is None:
            args.llm_max_samples = 50

    if not args.skip_ml:
        log.info("Running ML baselines (MNB, LR, RF, DT, KNN)...")
        run_ml_baselines(args.data_dir, output_dir)

    if not args.skip_nn:
        log.info("Running NN baselines (RNN, LSTM, GRU)...")
        run_nn_baselines(args.data_dir, output_dir)

    if not args.skip_zero_shot:
        log.info("Running zero-shot transformers (BART, DeBERTa)...")
        run_zero_shot(args.data_dir, output_dir)

    if not args.skip_llm:
        prompt_types = (
            ["basic", "few_shot"] if args.llm_prompt == "both" else [args.llm_prompt]
        )
        for prompt_type in prompt_types:
            log.info("Running LLM experiments (%s prompt)...", prompt_type)
            run_llm_experiments(
                args.data_dir,
                output_dir,
                prompt_type=prompt_type,
                backend=args.llm_backend,
                model_name=args.llm_model,
                local_model=args.llm_local_model,
                max_samples=args.llm_max_samples,
            )

    log.info("Writing summary tables...")
    summary_path = write_summary(output_dir)
    log.info("Pipeline complete. Results saved to: %s", output_dir)
    log.info("Summary report: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
