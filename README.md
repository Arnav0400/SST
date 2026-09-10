# Main Qwen3-8B reproduction

Each method has its own config and public trigger. There is no matrix launcher
and no runtime dependency on the repository's production training scripts.

```text
paper_reproduction/qwen3/
├── sst/{config.json,train.py}
├── baselines/
│   ├── sft/{config.json,train.py}
│   ├── sdft/{config.json,train.py}
│   ├── oplora/{config.json,paper_reference.json,train.py}
│   └── sculpting/{config.json,paper_reference.json,train.py}
├── common/                 # local data and Trainer utilities
├── data/                   # four exact, Git-LFS-backed dataset snapshots
├── workers/                # local DeepSpeed training implementations
├── deepspeed/              # local ZeRO-2 and ZeRO-3 configs
└── vendor/sdft/            # Apache-2.0 SDFT trainer copy
```

The public `train.py` files only resolve a method config and launch its local
worker. `_shared.py` keeps the matched Qwen3 optimizer recipe identical and
writes `reproduction_manifest.json` before training.

## Paper fidelity versus matched comparison

The main commands below are a **matched Qwen3 comparison**, not a claim that the
original baseline papers used the same optimizer recipe. Method-specific settings
follow the papers; model, data and optimizer budget follow this Qwen3 table.

### OPLoRA

[`baselines/oplora/config.json`](paper_reproduction/qwen3/baselines/oplora/config.json) uses the paper's
high-preservation OPLoRA-128 variant:

- absolute projection rank `k=128`, not a 99% rank fraction;
- LoRA rank/alpha 32;
- LoRA dropout 0.05;
- targets `q_proj`, `v_proj`, `up_proj`, `down_proj`, `o_proj`.

The OPLoRA paper also reports `k=16`. Its original experiments used one epoch,
batch 32, warmup ratio 0.03 and learning rates `2e-4` (commonsense/Python) or
`3e-4` (math); math additionally used LoRA rank/alpha 64. Those values are
recorded exactly in
[`paper_reference.json`](paper_reproduction/qwen3/baselines/oplora/paper_reference.json). The Qwen3 main
comparison intentionally keeps one fixed rank/alpha and the shared optimizer
recipe across all four datasets.

Source: [OPLoRA, arXiv:2510.13003](https://arxiv.org/abs/2510.13003).

### Sculpting

[`baselines/sculpting/config.json`](paper_reproduction/qwen3/baselines/sculpting/config.json) uses the
paper's main adaptive method:

- minimum retention ratio `mrr=0.1`;
- target retention ratio `trr=0.8`;
- activation-based per-layer rank allocation;
- projected gradients over `q`, `k`, `v`, `o`, `gate`, `up`, and `down`.

It does not use the earlier uniform 1% unfreeze ablation or a post-step parameter
projection. `calibration_samples: null` means the layer-importance estimate uses
the full training dataset, matching the paper's definition over `N` samples.

The paper's LLaMA-2-7B runs used one epoch, effective batch 8, learning rate
`1e-5`, weight decay 0.01 and AdamW betas `(0.9, 0.999)`. Those values are
recorded in
[`paper_reference.json`](paper_reproduction/qwen3/baselines/sculpting/paper_reference.json).

Source: [Sculpting Subspaces, arXiv:2504.07097](https://arxiv.org/abs/2504.07097).

## SST interface

[`sst/config.json`](paper_reproduction/qwen3/sst/config.json) contains exactly one method hyperparameter:

```json
{"rank_percent": 99}
```

The SST worker directly implements the applied-update projection. There is no
`constraint_mode`, `proj_update`, refresh flag, or alternate SST mode.

## Shared Qwen3 hyperparameters

Every main trigger emits this same optimizer recipe:

| Hyperparameter | Value |
|---|---:|
| Model | `Qwen/Qwen3-8B` |
| Learning rate | `5e-5` |
| Epochs | `2` |
| Global optimizer batch | `16` |
| Per-device micro-batch | `1` |
| Warmup | `10` optimizer steps |
| Scheduler | cosine |
| Weight decay | `0` |
| Gradient clipping | `1` |
| Precision | bf16 |
| Sequence length | `4096` |
| Seed | `42` |

Eight GPUs therefore use gradient accumulation two. Changing `--num-gpus`
recomputes accumulation to preserve global batch 16, or fails if that is not
possible exactly.

## Datasets

| Short name | Trainer dataset ID | Default path |
|---|---|---|
| `science` | `qwen3_science` | `paper_reproduction/qwen3/data/qwen3_science/train_data` |
| `medicine` | `qwen3_medicine` | `paper_reproduction/qwen3/data/qwen3_medicine/train_data` |
| `pathology` | `qwen3_pathology` | `paper_reproduction/qwen3/data/qwen3_pathology/train_data` |
| `finqa` | `qwen3_finqa` | `paper_reproduction/qwen3/data/qwen3_finqa/train_data` |

These exact snapshots are included with the artifact through Git LFS. They
contain 31,995 examples and occupy 153 MiB before Git transport compression.
Generated Hugging Face cache files and duplicate source JSONL files are not
included. See [`data/README.md`](paper_reproduction/qwen3/data/README.md) for checksums and provenance
notes.

For another dataset with the same `messages`/`output_text` schema, pass both a
name and its saved Hugging Face dataset directory:

```bash
python paper_reproduction/qwen3/sst/train.py \
  --dataset my_dataset --data-path /path/to/train_data
```

## Environment

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -r paper_reproduction/qwen3/requirements.txt
python paper_reproduction/qwen3/verify_data.py
```

## Trigger one training run

```bash
# SST
python paper_reproduction/qwen3/sst/train.py --dataset science

# Vanilla SFT
python paper_reproduction/qwen3/baselines/sft/train.py --dataset science

# SDFT
python paper_reproduction/qwen3/baselines/sdft/train.py --dataset science

# OPLoRA-128
python paper_reproduction/qwen3/baselines/oplora/train.py --dataset science

# Adaptive Sculpting
python paper_reproduction/qwen3/baselines/sculpting/train.py --dataset science
```

Replace `science` with `medicine`, `pathology`, or `finqa`. Add `--dry-run` to
inspect the exact local DeepSpeed worker command, and `--report-to wandb` to
enable experiment tracking.

To run one method on all four datasets:

```bash
for DATASET in science medicine pathology finqa; do
  python paper_reproduction/qwen3/sst/train.py --dataset "$DATASET"
done
```

Outputs default to `paper_runs/qwen3/<dataset>/<method>/`.

## Verify

```bash
python -m unittest paper_reproduction.qwen3.test_triggers
```

The tests verify all dataset aliases, the shared Qwen3 optimizer recipe, the
paper method settings, SST's one-parameter interface, and that every worker and
DeepSpeed config and default dataset resolves inside
`paper_reproduction/qwen3`.
