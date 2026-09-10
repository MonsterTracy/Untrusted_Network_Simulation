#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
vllm_conda_hook="$("${CONDA_EXE:?initialize Conda in the operator shell first}" shell.bash hook)"
eval "$vllm_conda_hook"
conda activate /data/yuxiao/envs/untrusted-network-simulation-vllm
exec vllm serve --config configs/deployment/qwen35-9b.yaml
