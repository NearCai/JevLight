#!/usr/bin/env bash
# serve_jev.sh — start a local Jev-style decision-model service for JevLight.
#
# The service speaks Jev's System One wire format (POST /v1/systemone), so
# JevLight connects with:
#   python run_jevlight.py --jev_transport local \
#       --jev_base_url http://127.0.0.1:8123
#
# Backends:
#   vllm  (default) Two-layer deployment (the "Jev-vLLM" route):
#                   1. vLLM serves an open causal-LM checkpoint (Tev1, Laya,
#                      Qwen, ... — any Hugging Face repo id) on an
#                      OpenAI-compatible endpoint (GPU).
#                   2. jevlight.local_server adapts it to /v1/systemone by
#                      reading option probabilities from next-token logprobs
#                      (one prefill per question, no decoding).
#   mock             Same /v1/systemone API with deterministic max-pressure
#                   answers — end-to-end pipeline runs without a GPU or key.
#   rbdash           RBDash-Decision image classifier (separate model env).
#
# Usage:
#   scripts/serve_jev.sh --backend mock
#   scripts/serve_jev.sh --backend rbdash --port 8125
#   scripts/serve_jev.sh --model tev1                     # alias, see below
#   scripts/serve_jev.sh --model <org>/<checkpoint>       # any HF repo id
#   scripts/serve_jev.sh --model tev1 --dry-run           # print plan only
#
# Aliases (edit or pass a full repo id to override):
#   tev1 -> togethercomputer/Tev1-0.8B-experimental
# For Laya & friends, pass the checkpoint id from its Hugging Face page.
#
# Environment overrides: PORT, VLLM_PORT, HOST, VENV_DIR, PYTHON,
# EXTRA_VLLM_ARGS (e.g. "--gpu-memory-utilization 0.85"), CUDA_VISIBLE_DEVICES
# (passed through to vLLM).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

BACKEND="vllm"
MODEL="${MODEL:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8123}"
VLLM_PORT="${VLLM_PORT:-8124}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-jev}"
PYTHON="${PYTHON:-python3}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
DRY_RUN=0

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --vllm-port) VLLM_PORT="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done

resolve_model() {
  case "$1" in
    tev1) echo "togethercomputer/Tev1-0.8B-experimental" ;;
    *) echo "$1" ;;
  esac
}

wait_http_ok() {
  # wait_http_ok <url> <timeout_seconds>
  local url="$1" timeout_s="$2" waited=0
  while true; do
    if "$PYTHON" -c "
import sys, urllib.request
sys.exit(0 if urllib.request.urlopen('$url', timeout=5).status == 200 else 1)
" 2>/dev/null; then
      return 0
    fi
    if [ "$waited" -ge "$timeout_s" ]; then
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

VLLM_PID=""
cleanup() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve_jev] stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

echo "[serve_jev] backend=$BACKEND adapter=http://$HOST:$PORT/v1/systemone"

if [ "$BACKEND" = "mock" ]; then
  # GPU-free pipeline smoke test: the adapter alone, deterministic answers.
  cd "$REPO_ROOT"
  run "$PYTHON" -m jevlight.local_server --backend mock \
    --host "$HOST" --port "$PORT" --model jev-mock
  exit 0
fi

if [ "$BACKEND" = "rbdash" ]; then
  MODEL_REPO=$(resolve_model "${MODEL:-RBDash-Team/RBDash-Decision-v0.2}")
  cd "$REPO_ROOT"
  run "$PYTHON" -m jevlight.local_server --backend rbdash \
    --host "$HOST" --port "$PORT" --model "$MODEL_REPO"
  exit 0
fi

# --- vllm backend -------------------------------------------------------- #

if [ -z "$MODEL" ]; then
  echo "error: --model is required for the vllm backend" >&2
  echo "  e.g. --model tev1   or   --model <org>/<checkpoint>" >&2
  exit 1
fi
MODEL_REPO=$(resolve_model "$MODEL")
SERVED_NAME=$(basename "$MODEL_REPO")
VLLM_URL="http://$HOST:$VLLM_PORT/v1"
LOG_DIR="$REPO_ROOT/logs/jev-serve"

echo "[serve_jev] checkpoint=$MODEL_REPO served-model-name=$SERVED_NAME"

if "$PYTHON" -c "
import sys, urllib.request
sys.exit(0 if urllib.request.urlopen('$VLLM_URL/models', timeout=5).status == 200 else 1)
" 2>/dev/null; then
  echo "[serve_jev] reusing vLLM already listening on $VLLM_URL"
else
  # Dedicated venv: vLLM pins its own torch build; keep it out of the
  # project environment.
  if [ ! -x "$VENV_DIR/bin/vllm" ]; then
    echo "[serve_jev] creating venv and installing vllm in $VENV_DIR (once)"
    run "$PYTHON" -m venv "$VENV_DIR"
    run "$VENV_DIR/bin/pip" install --upgrade pip
    run "$VENV_DIR/bin/pip" install "vllm>=0.11"
  fi
  echo "[serve_jev] starting vLLM on $VLLM_URL (logs: $LOG_DIR/vllm.log)"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $VENV_DIR/bin/vllm serve $MODEL_REPO --host $HOST \
--port $VLLM_PORT --served-model-name $SERVED_NAME $EXTRA_VLLM_ARGS"
  else
    mkdir -p "$LOG_DIR"
    # shellcheck disable=SC2086  EXTRA_VLLM_ARGS is a word-split flag list
    "$VENV_DIR/bin/vllm" serve "$MODEL_REPO" \
      --host "$HOST" --port "$VLLM_PORT" \
      --served-model-name "$SERVED_NAME" \
      $EXTRA_VLLM_ARGS >> "$LOG_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" > "$LOG_DIR/vllm.pid"
    # Generous timeout: first runs download the checkpoint.
    echo "[serve_jev] waiting for vLLM to become healthy (weights load can take minutes)..."
    if ! wait_http_ok "$VLLM_URL/models" 900; then
      echo "error: vLLM not healthy after 900s — see $LOG_DIR/vllm.log" >&2
      exit 1
    fi
  fi
fi

echo "[serve_jev] starting the /v1/systemone adapter on $HOST:$PORT"
echo "[serve_jev] once ready, run JevLight against it with:"
echo "  python run_jevlight.py --jev_transport local \\"
echo "      --jev_base_url http://$HOST:$PORT --jev_model $SERVED_NAME"
cd "$REPO_ROOT"
run "$PYTHON" -m jevlight.local_server --backend vllm \
  --host "$HOST" --port "$PORT" \
  --model "$SERVED_NAME" \
  --vllm_base_url "$VLLM_URL" --vllm_model "$SERVED_NAME"
