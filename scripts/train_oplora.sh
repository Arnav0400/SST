#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: scripts/train_oplora.sh DATASET [OPTIONS]}"
shift
exec python -m sst --method oplora --dataset "$dataset" "$@"
