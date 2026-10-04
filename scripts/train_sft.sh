#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: scripts/train_sft.sh DATASET [OPTIONS]}"
shift
exec python -m sst --method sft --dataset "$dataset" "$@"
