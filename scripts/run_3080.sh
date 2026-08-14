#!/usr/bin/env bash
# KVCIS-3 pipeline (3-class probe: fp16 / int8 / never-store) for a 10GB RTX 3080.
#
# Usage:
#   ./run_3080.sh          # quick pass (~10-15 min)
#   ./run_3080.sh --full   # larger run: better probe
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")/code"
REPO_PARENT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PY="$REPO_PARENT/.venv/Scripts/python.exe"          # Windows venv layout
[ -x "$PY" ] || PY="$REPO_PARENT/.venv/bin/python"  # POSIX venv layout
[ -x "$PY" ] || { echo "venv python not found; see kvcis/README_3080.md"; exit 1; }

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TRANSFORMERS_VERBOSITY=error

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
LAYER=10

if [ "${1:-}" = "--full" ]; then
    NPROMPTS=500; GENSTEPS=40; NTEXTS=40
else
    NPROMPTS=200; GENSTEPS=30; NTEXTS=20
fi

cd "$CODE_DIR"

echo "=== Step 1: single-prompt sanity check (trajectory + 3-class labels) ==="
"$PY" step1_single_prompt.py --model "$MODEL" --extraction-layer "$LAYER"

echo "=== Step 2: collect two-probe training data ($NPROMPTS prompts, horizon $GENSTEPS) ==="
"$PY" step2_collect_data.py --model "$MODEL" --extraction-layer "$LAYER" \
    --n-prompts "$NPROMPTS" --generation-steps "$GENSTEPS" \
    --evict-window 10 --max-prompt-tokens 96 --output-dir ../data

echo "=== Step 3: train probe A (original KVCIS regression) + probe B (evictability) ==="
"$PY" step3_train_probe.py --data-dir ../data --output-dir ../data/probe2

echo "=== Step 4: deferred-eviction eval (KVCIS / KVCIS+Evict vs Baseline & INT8) ==="
"$PY" step4_compression_eval.py --model "$MODEL" --extraction-layer "$LAYER" \
    --probe-path ../data/probe2 --n-texts "$NTEXTS"

echo "All done. Results: ../results/kvcis2p_results.json"
