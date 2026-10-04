# SST

Official implementation of **SST**, a subspace-constrained fine-tuning method
for updating language models while limiting movement along protected weight
directions.

The repository includes SST, four commonly used fine-tuning baselines, a
Qwen3-8B configuration, and four ready-to-run datasets. The layout follows a
simple method/config/script split:

```text
SST/
├── sst/                    # library and training implementations
│   ├── cli.py
│   ├── data.py
│   ├── training.py
│   ├── methods/            # sst, sft, sdft, oplora, sculpting
│   └── vendor/sdft/
├── configs/
│   ├── model/qwen3-8b.yaml
│   ├── data/{science,medicine,pathology,finqa}.yaml
│   ├── method/{sst,sft,sdft,oplora,sculpting}.yaml
│   └── deepspeed/
├── scripts/                # one launcher per method
├── data/                   # Git-LFS-backed datasets
└── tests/
```

## Installation

```bash
git lfs install
git lfs pull

python -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/verify_data.py
```

## Training

Each method has its own launcher. The first argument is the dataset:

```bash
# SST
bash scripts/train_sst.sh science

# Baselines
bash scripts/train_sft.sh science
bash scripts/train_sdft.sh science
bash scripts/train_oplora.sh science
bash scripts/train_sculpting.sh science
```

Replace `science` with `medicine`, `pathology`, or `finqa`.

The equivalent Python interface is:

```bash
python -m sst --method sst --dataset science
python -m sst --method sft --dataset medicine
```

Useful options:

```bash
# Inspect the resolved DeepSpeed command.
bash scripts/train_sst.sh science --dry-run

# Use four GPUs while preserving global batch size 16.
bash scripts/train_sst.sh science --num-gpus 4

# Enable Weights & Biases logging.
bash scripts/train_sst.sh science --report-to wandb

# Select an output directory.
bash scripts/train_sst.sh science --output-dir runs/my_sst_run
```

Runs are written to `runs/<dataset>/<method>/` by default. Each run records its
resolved settings and launch command in `run_config.json`.

## Configuration

Configuration is split by concern, following the same composition pattern used
by the launch scripts:

- [`configs/model/qwen3-8b.yaml`](configs/model/qwen3-8b.yaml) contains the
  shared model and optimizer settings.
- [`configs/data/`](configs/data) maps short dataset names to local paths.
- [`configs/method/`](configs/method) contains only method-specific settings.
- [`configs/deepspeed/`](configs/deepspeed) contains the distributed runtime
  settings.

SST intentionally exposes one method hyperparameter:

```yaml
rank_percent: 99
```

The default Qwen3-8B settings are shared across all methods:

| Setting | Value |
|---|---:|
| Model | `Qwen/Qwen3-8B` |
| Learning rate | `5e-5` |
| Epochs | `2` |
| Global batch size | `16` |
| Micro-batch size | `1` |
| Warmup steps | `10` |
| Scheduler | cosine |
| Weight decay | `0` |
| Max sequence length | `4096` |
| Precision | bf16 |
| Seed | `42` |

## Datasets

| Name | Examples | Config |
|---|---:|---|
| Science | 9,000 | [`science.yaml`](configs/data/science.yaml) |
| Medicine | 8,996 | [`medicine.yaml`](configs/data/medicine.yaml) |
| Pathology | 8,999 | [`pathology.yaml`](configs/data/pathology.yaml) |
| FinQA | 5,000 | [`finqa.yaml`](configs/data/finqa.yaml) |

To train on another saved Hugging Face dataset with `messages` and
`output_text` columns:

```bash
python -m sst \
  --method sst \
  --dataset my_data \
  --data-path /path/to/my_data
```

Use `--no-thinking` if the dataset should use the non-thinking chat template.

## Baselines

The included baselines are vanilla SFT, SDFT, OPLoRA, and adaptive Sculpting.
Their public settings live in separate files under
[`configs/method/`](configs/method); the implementations live beside SST under
[`sst/methods/`](sst/methods).

## Tests

```bash
python -m unittest discover -s tests
```

The tests validate config composition, all method/dataset combinations,
SST's one-parameter interface, global-batch preservation, and the public launch
scripts.
