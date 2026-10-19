# Commands

Each shell launcher accepts a dataset name followed by the shared CLI options.

```bash
bash scripts/train_<method>.sh <dataset> [options]
```

## Launchers

| Method | Command |
|---|---|
| SST | `bash scripts/train_sst.sh science` |
| SFT | `bash scripts/train_sft.sh science` |
| SDFT | `bash scripts/train_sdft.sh science` |
| OPLoRA | `bash scripts/train_oplora.sh science` |
| Sculpting | `bash scripts/train_sculpting.sh science` |

Available bundled datasets are `science`, `finqa`, `pathology`, and `medicine`.

## Useful options

```bash
# Inspect the complete DeepSpeed command without training.
bash scripts/train_sst.sh science --dry-run

# Use four GPUs while preserving global batch size 16.
bash scripts/train_sst.sh science --num-gpus 4

# Choose the output location and run label.
bash scripts/train_sst.sh science \
  --output-dir runs/experiments/sst_science \
  --run-name sst_science_seed42

# Enable Weights & Biases reporting.
bash scripts/train_sst.sh science --report-to wandb

# Override chat-template thinking for a dataset.
bash scripts/train_sst.sh science --no-thinking
```

The same interface is available through Python or the installed console
script:

```bash
python -m sst --method sst --dataset science
sst-train --method sft --dataset medicine
```

Run `python -m sst --help` for all shared options.

## Outputs

Runs default to:

```text
runs/<dataset>/<method>/
```

Before training starts, the launcher writes `run_config.json` with the resolved
configuration, exact command, config hashes, GPU count, and Git commit. Final
model artifacts are written beneath the run directory by the selected trainer.

## Data check

The bundled Arrow files are stored through Git LFS. Verify them before a long
run:

```bash
git lfs pull
python scripts/verify_data.py
```

