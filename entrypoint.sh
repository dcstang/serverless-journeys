#!/usr/bin/env bash
# Entrypoint for the self-contained GPU job image (Dockerfile.gpu).
#
# Starts a local vLLM OpenAI-compatible server, waits for it to finish
# loading the model onto the GPU, runs the generation pipeline against it,
# then shuts the server down. This lets a single Nebius GPU job do model
# serving + generation in one process tree instead of needing a standing
# inference endpoint plus a separate job.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"

echo "[entrypoint] Starting vLLM server with model: ${MODEL}"
# shellcheck disable=SC2086
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

cleanup() {
    echo "[entrypoint] Shutting down vLLM server (pid ${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[entrypoint] Waiting up to ${VLLM_STARTUP_TIMEOUT}s for vLLM server to become ready..."
ready=false
for ((i = 0; i < VLLM_STARTUP_TIMEOUT; i++)); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[entrypoint] vLLM server ready after ${i}s"
        ready=true
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[entrypoint] vLLM server process died during startup" >&2
        exit 1
    fi
    sleep 1
done

if [ "${ready}" != "true" ]; then
    echo "[entrypoint] Timed out waiting for vLLM server after ${VLLM_STARTUP_TIMEOUT}s" >&2
    exit 1
fi

export LLM_PROVIDER=nebius
export NEBIUS_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export NEBIUS_API_KEY="${NEBIUS_API_KEY:-local-vllm-no-key-required}"
export MODEL

echo "[entrypoint] Running generation pipeline against local vLLM server..."
python3 main.py "$@"
