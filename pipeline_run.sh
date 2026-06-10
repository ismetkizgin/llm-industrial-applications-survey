#!/usr/bin/env bash
# End-to-end reproduction pipeline for:
# "Smart Expert System: Large Language Models as Text Classifiers"
# https://arxiv.org/abs/2405.10523
# Reference code: https://github.com/yeyimilk/llm-zero-shot-classifiers

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/results/run_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

# Load .env if present (OPENAI_API_KEY, LLM_* settings, etc.)
ENV_LOADED=0
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
  ENV_LOADED=1
fi

OPENAI_MODEL="${OPENAI_MODEL:-gpt-3.5-turbo}"
LLM_BACKEND="${LLM_BACKEND:-auto}"

# Modes:
#   full   - all experiments (ML, NN, zero-shot, LLM)
#   fast   - ML + NN only, quick smoke test
#   paper  - ML + NN + zero-shot (no API key required)
MODE="${1:-paper}"

mkdir -p "${OUTPUT_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== Smart Expert System Pipeline ==="
if [[ "${ENV_LOADED}" -eq 1 ]]; then
  log "Loaded configuration from .env"
else
  log "No .env file found — copy .env.example to .env to configure API keys"
fi
log "Mode: ${MODE}"
log "Output: ${OUTPUT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating virtual environment..."
  python3 -m venv "${VENV_DIR}"
fi

log "Installing dependencies..."
"${PYTHON}" -m pip install --upgrade pip setuptools wheel >> "${LOG_FILE}" 2>&1
"${PYTHON}" -m pip install -r "${ROOT_DIR}/requirements.txt" >> "${LOG_FILE}" 2>&1

# Download NLTK resources used by ML preprocessing
"${PYTHON}" -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" >> "${LOG_FILE}" 2>&1

PIPELINE_ARGS=(
  --output-dir "${OUTPUT_DIR}"
)

case "${MODE}" in
  full)
    log "Running FULL experiment (ML + NN + zero-shot + LLM)..."
    if [[ "${LLM_BACKEND}" == "openai" ]] || { [[ "${LLM_BACKEND}" == "auto" ]] && [[ -n "${OPENAI_API_KEY:-}" ]]; }; then
      log "Using OpenAI for LLM experiments (model: ${OPENAI_MODEL})."
      PIPELINE_ARGS+=(--llm-backend openai --llm-model "${OPENAI_MODEL}" --llm-prompt both)
    else
      log "LLM will use local HuggingFace model (set OPENAI_API_KEY in .env for OpenAI)."
      PIPELINE_ARGS+=(--llm-backend local --llm-prompt both)
    fi
    ;;
  fast)
    log "Running FAST smoke test (ML + NN only)..."
    PIPELINE_ARGS+=(--skip-zero-shot --skip-llm)
    ;;
  paper)
    log "Running PAPER reproduction (ML + NN + zero-shot, no API)..."
    PIPELINE_ARGS+=(--skip-llm)
    ;;
  llm-only)
    log "Running LLM experiments only..."
    PIPELINE_ARGS+=(--skip-ml --skip-nn --skip-zero-shot --llm-prompt both)
    if [[ "${LLM_BACKEND}" == "openai" ]] || { [[ "${LLM_BACKEND}" == "auto" ]] && [[ -n "${OPENAI_API_KEY:-}" ]]; }; then
      PIPELINE_ARGS+=(--llm-backend openai --llm-model "${OPENAI_MODEL}")
    else
      PIPELINE_ARGS+=(--llm-backend local --llm-max-samples "${LLM_MAX_SAMPLES:-50}")
    fi
    ;;
  *)
    echo "Usage: $0 [full|fast|paper|llm-only]"
    echo ""
    echo "  full     - complete pipeline including LLM (OpenAI if key set, else local)"
    echo "  fast     - ML + NN baselines only (quickest)"
    echo "  paper    - ML + NN + BART/DeBERTa (default, no API key)"
    echo "  llm-only - LLM prompting experiments only"
    exit 1
    ;;
esac

log "Starting Python pipeline..."
"${PYTHON}" "${ROOT_DIR}/src/run_pipeline.py" "${PIPELINE_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

log "=== Pipeline finished successfully ==="
log "Results directory: ${OUTPUT_DIR}"
log "Summary report:    ${OUTPUT_DIR}/RESULTS.md"
log "JSON summary:      ${OUTPUT_DIR}/summary.json"
