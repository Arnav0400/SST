# Configuration

Training configuration is split by concern so paired methods can share the
same model and data recipe.

```text
configs/
├── model/qwen3-8b.yaml
├── data/{science,finqa,pathology,medicine}.yaml
├── method/{sst,sft,sdft,oplora,sculpting}.yaml
└── deepspeed/{zero2,zero3}.json
```

The launcher validates every file and records the resolved settings, command,
config paths, hashes, and Git commit in `run_config.json`.

## Default model recipe

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
| Maximum sequence length | `4096` |
| Precision | bf16 |
| Seed | `42` |

The launcher derives gradient accumulation from the global batch size, GPU
count, and micro-batch size. For example, the default recipe uses accumulation
2 on eight GPUs and accumulation 4 on four GPUs.

## Method settings

| File | Main controls |
|---|---|
| `method/sst.yaml` | protected singular-direction percentage and optional refresh interval |
| `method/sft.yaml` | no method-specific settings |
| `method/sdft.yaml` | generation, reference mix, and vLLM settings |
| `method/oplora.yaml` | LoRA and projection ranks, target matrices |
| `method/sculpting.yaml` | adaptive retention range and calibration |

See the [method guide](../sst/methods/README.md) for the behavior behind each
setting.

For SST, `refresh_every: null` is the default fixed-basis mode. Set a positive
integer to refresh after that many successful optimizer updates, or override
it with `--refresh-every 100`. Old rank-only SST configurations remain compatible.
On resume, the checkpoint's active basis and refresh schedule take precedence.

## Dataset settings

Each dataset config provides a short name, the local saved-dataset path, and
whether to enable the model's thinking chat template:

```yaml
dataset_id: qwen3_science
path: data/qwen3_science/train_data
thinking: true
```

To use another saved Hugging Face dataset with `messages` and `output_text`
columns:

```bash
python -m sst \
  --method sst \
  --dataset my_data \
  --data-path /path/to/my_data
```

Pass `--no-thinking` if the dataset should use the non-thinking chat template.

## Custom recipes

For a tracked experiment, copy the configuration tree, change the required
YAML or DeepSpeed files, and point the launcher to it:

```bash
python -m sst \
  --method sst \
  --dataset science \
  --config-dir /path/to/configs \
  --dry-run
```

The custom directory must retain the same `model/`, `data/`, `method/`, and
`deepspeed/` layout.

