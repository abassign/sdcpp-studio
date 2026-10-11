#!/usr/bin/env bash
#
# sdcpp-studio launcher
# ---------------------
#   ./go.sh                 start the web UI (builds the image if missing)
#   ./go.sh setup           download the demo-min model preset
#   ./go.sh download <id>   download a preset from models.yaml (see ./go.sh list)
#   ./go.sh list            list model presets
#   ./go.sh build           build the Docker image for the detected GPU
#   ./go.sh shell           interactive shell in the container
#   ./go.sh --stop          stop the server (frees GPU/VRAM)
#   ./go.sh --stop-all      stop every GPU container of this project
#   ./go.sh --help          show this help
#
# Env overrides: MODELS_DIR, OUTPUTS_DIR, HF_CACHE_DIR, APP_DIR, PORT,
#                GPU_FLAVOR (rocm|cuda|cpu), IMAGE_NAME, HF_TOKEN.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$SCRIPT_DIR}"

# Optional .env defaults (do NOT override variables already set in the env)
if [ -f "$APP_DIR/.env" ]; then
    while IFS='=' read -r _k _v; do
        case "$_k" in ''|\#*) continue ;; esac
        _k="$(printf '%s' "$_k" | tr -d '[:space:]')"
        [ -z "$_k" ] && continue
        if [ -z "${!_k:-}" ]; then
            _v="${_v%\"}"; _v="${_v#\"}"   # strip surrounding quotes
            export "$_k=$_v"
        fi
    done < "$APP_DIR/.env"
fi
PARENT_DIR="$(dirname "$APP_DIR")"
IMAGE_NAME="${IMAGE_NAME:-sdcpp-studio}"
PORT="${PORT:-8000}"
CONTAINER_NAME="sdcpp_studio"

# ---------------------------------------------------------------------------
# Data dirs (context aware: reuse ./models ./outputs from the container folder)
# ---------------------------------------------------------------------------
pick_dir() { # $1=env value  $2=parent folder name  $3=local folder name
    local envv="$1" pname="$2" lname="$3"
    if [ -n "$envv" ]; then echo "$envv"
    elif [ -d "$PARENT_DIR/$pname" ] && [ -n "$(ls -A "$PARENT_DIR/$pname" 2>/dev/null)" ]; then
        echo "$PARENT_DIR/$pname"
    else echo "$APP_DIR/$lname"; fi
}
MODELS_DIR="${MODELS_DIR:-$(pick_dir "" models models)}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$(pick_dir "" outputs outputs)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$PARENT_DIR/hf_cache}"
IN_MODELS="/app/models"; IN_OUTPUTS="/app/outputs"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
GPU_FLAVOR="${GPU_FLAVOR:-}"
if [ -z "$GPU_FLAVOR" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        GPU_FLAVOR="cuda"
    elif [ -e /dev/kfd ]; then
        GPU_FLAVOR="rocm"
    else
        GPU_FLAVOR="cpu"
    fi
fi
IMAGE="${IMAGE_NAME}:${GPU_FLAVOR}"

case "$GPU_FLAVOR" in
    rocm) GPU_TYPE_ENV="ROCm" ;;
    cuda) GPU_TYPE_ENV="CUDA" ;;
    *)    GPU_TYPE_ENV="CPU" ;;
esac

gpu_run_args() {
    case "$GPU_FLAVOR" in
        rocm) printf '%s\n' --device=/dev/kfd --device=/dev/dri --group-add video --group-add render ;;
        cuda) printf '%s\n' --gpus all ;;
        *)    : ;;
    esac
}

usage() {
    sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
EXTRA="${1:-}"
case "$EXTRA" in
    --help|-h|help) usage; exit 0 ;;
esac

# --stop / --stop-all -------------------------------------------------------
if [ "$EXTRA" = "--stop" ] || [ "$EXTRA" = "--stop-all" ]; then
    echo "Stopping '$CONTAINER_NAME' ..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm   "$CONTAINER_NAME" >/dev/null 2>&1 || true
    if [ "$EXTRA" = "--stop-all" ]; then
        for c in $(docker ps --format '{{.Names}}'); do
            if docker inspect "$c" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null \
                 | grep -qE '/dev/kfd|/dev/dri'; then
                echo "Stopping '$c' ..."
                docker stop "$c" >/dev/null 2>&1 || true
            fi
        done
    fi
    echo "Done."
    exit 0
fi

# build ---------------------------------------------------------------------
if [ "$EXTRA" = "build" ]; then
    BASE_ARG=""
    [ "$GPU_FLAVOR" = "cuda" ] && BASE_ARG="--build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04"
    echo "Building $IMAGE ($GPU_FLAVOR) ..."
    docker build --build-arg "GPU_FLAVOR=$GPU_FLAVOR" $BASE_ARG -t "$IMAGE" "$APP_DIR"
    echo "Done."
    exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image '$IMAGE' not found. Build it first:  ./go.sh build"
    exit 1
fi

# one-shot runner (installer / listing) -------------------------------------
run_oneshot() {
    # shellcheck disable=SC2046
    docker run --rm \
        $(gpu_run_args) \
        -e "HF_TOKEN=${HF_TOKEN:-}" \
        -e "MODELS_DIR=$IN_MODELS" \
        -v "$APP_DIR:/app" \
        -v "$MODELS_DIR:$IN_MODELS" \
        -v "$OUTPUTS_DIR:$IN_OUTPUTS" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        "$IMAGE" "$@"
}

if [ "$EXTRA" = "list" ]; then
    run_oneshot python3 /app/tools/download_models.py --list
    exit 0
fi
if [ "$EXTRA" = "setup" ]; then
    echo "Installing the demo-min model preset into: $MODELS_DIR"
    run_oneshot python3 /app/tools/download_models.py --preset demo-min --yes
    exit 0
fi
if [ "$EXTRA" = "download" ]; then
    PRESET="${2:-}"
    [ -z "$PRESET" ] && { echo "usage: ./go.sh download <preset>   (see ./go.sh list)"; exit 1; }
    run_oneshot python3 /app/tools/download_models.py --preset "$PRESET" --yes
    exit 0
fi

# shell ---------------------------------------------------------------------
if [ "$EXTRA" = "shell" ]; then
    # shellcheck disable=SC2046
    exec docker run --rm -it \
        $(gpu_run_args) --network host --ipc host \
        -e "PORT=$PORT" -e "GPU_FLAVOR=$GPU_FLAVOR" -e "GPU_TYPE=$GPU_TYPE_ENV" \
        -e "MODELS_DIR=$IN_MODELS" -e "OUTPUTS_DIR=$IN_OUTPUTS" \
        -v "$APP_DIR:/app" \
        -v "$MODELS_DIR:$IN_MODELS" \
        -v "$OUTPUTS_DIR:$IN_OUTPUTS" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        "$IMAGE" bash
fi

# default: start the server -------------------------------------------------
echo "sdcpp-studio"
echo "  GPU     : $GPU_FLAVOR ($IMAGE)"
echo "  Models  : $MODELS_DIR"
echo "  Outputs : $OUTPUTS_DIR"
echo "  HF cache: $HF_CACHE_DIR"
echo "  Port    : $PORT"

if [ -z "$(ls -A "$MODELS_DIR/checkpoints" 2>/dev/null)" ]; then
    echo ""
    echo "No models installed yet. Run './go.sh setup' to download the demo-min preset."
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2046
docker run -d --name "$CONTAINER_NAME" \
    $(gpu_run_args) --network host --ipc host \
    -e "PORT=$PORT" -e "GPU_FLAVOR=$GPU_FLAVOR" -e "GPU_TYPE=$GPU_TYPE_ENV" \
    -e "MODELS_DIR=$IN_MODELS" -e "OUTPUTS_DIR=$IN_OUTPUTS" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -v "$APP_DIR:/app" \
    -v "$MODELS_DIR:$IN_MODELS" \
    -v "$OUTPUTS_DIR:$IN_OUTPUTS" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    "$IMAGE" python3 server.py

echo ""
echo "Started. UI at http://localhost:$PORT   (stop: ./go.sh --stop)"
