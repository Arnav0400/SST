#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: scripts/train_sdft.sh DATASET [OPTIONS]}"
shift
exec python -m sst --method sdft --dataset "$dataset" "$@"
