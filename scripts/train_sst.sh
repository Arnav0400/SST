#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: scripts/train_sst.sh DATASET [OPTIONS]}"
shift
exec python -m sst --method sst --dataset "$dataset" "$@"
