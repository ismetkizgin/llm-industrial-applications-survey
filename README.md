# Smart Expert System — Text Classification Experiment Pipeline

Automated reproduction pipeline for [Wang et al. (2024) — *Smart Expert System: Large Language Models as Text Classifiers*](https://arxiv.org/abs/2405.10523).

Reference implementation: [yeyimilk/llm-zero-shot-classifiers](https://github.com/yeyimilk/llm-zero-shot-classifiers)

## Overview

This project compares LLM-based text classification against traditional ML, deep learning, and zero-shot transformer models on four public datasets. A single shell script runs the full workflow end to end and writes all outputs to a timestamped results folder.

**Evaluation metrics:** Accuracy (ACC), macro F1, and the paper's Uncertainty/Error Rate (U/E).

## Datasets

| Key | Description | Classes |
|-----|-------------|---------|
| `Corona_NLP_new` | COVID-19 tweet sentiment | 3 |
| `ecommerceDataset` | E-commerce product classification | 4 |
| `financial_sentiment` | Economic text sentiment | 3 |
| `sms_spam` | SMS spam detection | 2 |

Sampling strategy (paper Section V-A):

- Training set > 10,000 → stratified sample of 10,000
- Test set > 800 → stratified sample of 800

## Experiment Modules

| Module | Models | Input |
|--------|--------|-------|
| ML baseline | MNB, LR, RF, DT, KNN | Preprocessed + tokenized text |
| NN baseline | RNN, LSTM, GRU | Preprocessed + tokenized text |
| Zero-shot | BART-large-MNLI, DeBERTa-large-MNLI | Raw text |
| LLM (optional) | OpenAI GPT or local HuggingFace model | Raw text + prompt |

> **Note:** Fine-tuning (Llama-3, Qwen) requires a GPU and a separate training setup. It is not included in this pipeline. See the [reference repository](https://github.com/yeyimilk/llm-zero-shot-classifiers) for fine-tuning scripts.

---

## Running from Scratch

Follow these steps on a clean machine with no prior setup.

### 1. Prerequisites

Install the following before you begin:

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or newer (`python3 --version`) |
| **Git** | To clone the repository |
| **Disk space** | ~8 GB (dependencies, datasets, model cache) |
| **RAM** | 8 GB minimum; 16 GB recommended |
| **GPU** | Optional but recommended for NN and zero-shot stages |
| **Internet** | Required for downloading datasets and Python packages |

Check your Python version:

```bash
python3 --version
# Expected: Python 3.10.x or higher
```

### 2. Get the Project

**Option A — Clone from Git:**

```bash
git clone <your-repo-url> llm-industrial-applications-survey
cd llm-industrial-applications-survey
```

**Option B — Use an existing copy:**

```bash
cd /path/to/llm-industrial-applications-survey
```

### 3. Configure Environment Variables

Copy the example file and edit it with your settings:

```bash
cp .env.example .env
```

Open `.env` in your editor. For a first run without LLM API calls, you can leave it as-is. To use OpenAI for LLM experiments, set:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
LLM_BACKEND=auto
```

> `.env` is gitignored. Never commit API keys. Use `.env.example` as the shared template.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | For OpenAI LLM runs | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-3.5-turbo` | OpenAI chat model |
| `LLM_BACKEND` | No | `auto` | `auto`, `openai`, or `local` |
| `LLM_LOCAL_MODEL` | No | `TinyLlama/...` | HuggingFace model for local backend |
| `LLM_PROMPT` | No | `basic` | `basic`, `few_shot`, or `both` |
| `LLM_MAX_SAMPLES` | No | — | Cap LLM test samples per dataset |
| `HF_TOKEN` | No | — | HuggingFace token for gated models |
| `CUDA_VISIBLE_DEVICES` | No | — | GPU selection (`0`, `0,1`, or `-1` for CPU) |

Both `pipeline_run.sh` and `src/run_pipeline.py` load `.env` automatically.

### 4. Make the Script Executable

```bash
chmod +x pipeline_run.sh
```

### 5. Run the Pipeline

The script handles everything automatically: virtual environment creation, dependency installation, dataset download, experiment execution, and result aggregation.

**Recommended first run (no API key required):**

```bash
./pipeline_run.sh paper
```

**Quick smoke test (fastest, ML + NN only):**

```bash
./pipeline_run.sh fast
```

**Full run including LLM experiments:**

```bash
# Set OPENAI_API_KEY in .env first, then:
./pipeline_run.sh full
```

### 6. What Happens During a Run

When you execute `pipeline_run.sh`, the following steps run in order:

1. **Create virtual environment** — `.venv/` in the project root
2. **Install dependencies** — packages from `requirements.txt`
3. **Download NLTK data** — stopwords and tokenizers for text preprocessing
4. **Download datasets** — CSV files from the reference GitHub repository into `data/`
5. **Prepare mini datasets** — stratified 10k train / 800 test splits
6. **Run experiments** — depending on the selected mode
7. **Write summary** — `RESULTS.md` and `summary.json` in the output folder

Expected runtime (approximate, CPU):

| Mode | Duration |
|------|----------|
| `fast` | 15–45 min |
| `paper` | 1–3 hours |
| `full` | 2–5+ hours (LLM stage depends on backend) |

### 7. Verify the Results

After a successful run, the script prints the output path:

```
Results directory: results/run_20260610_120000/
Summary report:    results/run_20260610_120000/RESULTS.md
```

Check that key files exist:

```bash
ls results/run_*/
# Expected: run_config.json, pipeline.log, summary.json, RESULTS.md
#           ml_baselines/, nn_baselines/, zero_shot/ (and llm/ if applicable)
```

Open the summary report:

```bash
cat results/run_*/RESULTS.md
```

### 8. Manual Setup (Alternative to the Shell Script)

If you prefer to set up the environment yourself:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Run the pipeline (.env is loaded automatically)
python src/run_pipeline.py --output-dir results/my_run
```

### 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| `Permission denied: ./pipeline_run.sh` | Run `chmod +x pipeline_run.sh` |
| `python3: command not found` | Install Python 3.10+ or use `python` instead |
| TensorFlow install fails | Use Python 3.10–3.12; TF may not support the latest Python yet |
| Out of memory during NN/zero-shot | Use `./pipeline_run.sh fast` or add `--skip-nn --skip-zero-shot` |
| Dataset download fails | Check internet connection; retry without `--skip-download` |
| `OPENAI_API_KEY` errors | Set it in `.env` or use `LLM_BACKEND=local` in `.env` |
| `.env` not loaded | Ensure the file is in the project root (same folder as `pipeline_run.sh`) |
| Slow first run | Normal — models and datasets are downloaded on first use |

Clean reinstall:

```bash
rm -rf .venv data results
./pipeline_run.sh fast
```

---

## Run Modes

```bash
./pipeline_run.sh paper      # Default — paper reproduction (no API key)
./pipeline_run.sh fast       # Quick test — ML + NN only
./pipeline_run.sh full       # Full experiment — includes LLM
./pipeline_run.sh llm-only   # LLM prompting experiments only
```

| Mode | ML | NN | Zero-shot | LLM |
|------|:--:|:--:|:---------:|:---:|
| `fast` | ✓ | ✓ | — | — |
| `paper` | ✓ | ✓ | ✓ | — |
| `full` | ✓ | ✓ | ✓ | ✓ |
| `llm-only` | — | — | — | ✓ |

## Advanced Usage (Python CLI)

After activating the virtual environment:

```bash
source .venv/bin/activate
python src/run_pipeline.py --help
```

Examples:

```bash
# ML baselines only
python src/run_pipeline.py --skip-nn --skip-zero-shot --skip-llm

# LLM few-shot with a sample limit
python src/run_pipeline.py \
  --skip-ml --skip-nn --skip-zero-shot \
  --llm-prompt few_shot \
  --llm-max-samples 50 \
  --llm-backend local

# Re-run experiments on cached data (skip download)
python src/run_pipeline.py --skip-download --output-dir results/rerun_01
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--output-dir PATH` | Output directory (default: `results/run_<timestamp>`) |
| `--data-dir PATH` | Dataset directory (default: `data/`) |
| `--skip-download` | Skip dataset download |
| `--skip-ml` / `--skip-nn` / `--skip-zero-shot` / `--skip-llm` | Skip a stage |
| `--llm-backend auto\|openai\|local` | LLM inference backend |
| `--llm-prompt basic\|few_shot\|both` | Prompt strategy |
| `--llm-max-samples N` | Limit LLM test samples per dataset |
| `--fast` | Skip zero-shot; cap LLM at 50 samples |

## Output Structure

```
results/run_20260610_120000/
├── run_config.json          # Run configuration
├── pipeline.log             # Shell script log
├── summary.json             # Structured summary
├── RESULTS.md               # Paper-style result tables
├── ml_baselines/
│   ├── Corona_NLP_new.json
│   ├── ecommerceDataset.json
│   ├── financial_sentiment.json
│   └── sms_spam.json
├── nn_baselines/
│   └── ...
├── zero_shot/
│   └── ...
└── llm/
    └── openai/ or local/
        └── basic/ or few_shot/
            └── ...
```

Each JSON file contains `metrics` (accuracy, f1_macro, ue_rate) and `predictions`.

## Project Structure

```
.
├── pipeline_run.sh          # Main entry point
├── requirements.txt
├── README.md
├── .env.example             # Environment variable template (copy to .env)
├── .env                     # Local secrets and config (gitignored)
├── data/                    # Downloaded datasets (gitignored)
├── results/                 # Experiment outputs (gitignored)
└── src/
    ├── run_pipeline.py      # Orchestrator
    ├── config.py            # Dataset and sampling constants
    ├── summarize.py         # Result tables
    ├── data/
    │   ├── download.py      # Download from reference repo
    │   ├── prepare.py       # Stratified mini-set creation
    │   └── loader.py        # Data loading
    ├── experiments/
    │   ├── ml_baselines.py
    │   ├── nn_baselines.py
    │   ├── zero_shot.py
    │   └── llm_classifier.py
    ├── prompts/             # Prompt templates from the paper
    └── utils/
        ├── metrics.py       # ACC, F1, U/E rate
        ├── io.py
        └── text_clean.py
```

## U/E Rate (Uncertainty/Error Rate)

Reliability metric proposed in the paper:

$$\text{U/E} = \frac{U + E}{N}$$

- **U** — Uncertain outputs (e.g. refusal to classify)
- **E** — Erroneous outputs (invalid label, parse failure, hallucination)
- **N** — Total number of test samples

Computed only for LLM experiments. ML and NN models are deterministic, so U/E is not reported for them.

## Citation

If you use this work, please cite the original paper:

```bibtex
@article{wang2024smart,
  title={Smart Expert System: Large Language Models as Text Classifiers},
  author={Wang, Zhiqiang and Pang, Yiran and Lin, Yanbin},
  journal={arXiv preprint arXiv:2405.10523},
  year={2024}
}
```

Related prior work:

```bibtex
@article{wang2023large,
  title={Large Language Models Are Zero-Shot Text Classifiers},
  author={Wang, Zhiqiang and Pang, Yiran and Lin, Yanbin},
  journal={arXiv preprint arXiv:2312.01044},
  year={2023}
}
```

## License

This reproduction project is intended for education and research. See the [reference repository](https://github.com/yeyimilk/llm-zero-shot-classifiers) for dataset and original code license terms.
