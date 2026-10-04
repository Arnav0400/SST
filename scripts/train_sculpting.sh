#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: scripts/train_sculpting.sh DATASET [OPTIONS]}"
shift
exec python -m sst --method sculpting --dataset "$dataset" "$@"
