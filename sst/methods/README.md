# Methods

All methods share the same model, dataset, and run interface. Method-specific
settings live in [`configs/method/`](../../configs/method), and each public
launcher delegates to the matching implementation in this directory.

| Method | What is trained | Constraint | DeepSpeed |
|---|---|---|---|
| SST | Full model | Applied update stays outside principal weight directions | ZeRO-2 |
| SFT | Full model | None | ZeRO-2 |
| SDFT | Student from its generated targets | On-policy self-distillation | ZeRO-3 |
| OPLoRA | Low-rank adapters | Adapter update is projected on both sides | ZeRO-2 |
| Sculpting | Full model | Layer-adaptive gradient projection | ZeRO-2 |

## SST

SST computes the principal left and right singular directions of each targeted
weight matrix at the start of training. After every optimizer step, it projects
the cumulative change from the reference weights into the free-to-free
subspace.

Targeted matrices are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`.

```yaml
# configs/method/sst.yaml
rank_percent: 99
```

`rank_percent` is the percentage of singular directions protected on each side.
It must be between 0 and 100. The public launcher currently implements the
fixed-basis, no-refresh SST variant.

```bash
bash scripts/train_sst.sh science
```

## SFT

Vanilla full-parameter supervised fine-tuning. It has no method-specific
settings and serves as the direct baseline for SST.

```bash
bash scripts/train_sft.sh science
```

## SDFT

Self-distillation fine-tuning generates targets from a synchronized reference
model and trains a student on those generations. It uses the vendored
`DistilTrainer` implementation and colocated vLLM by default.

Important settings:

| Setting | Default | Meaning |
|---|---:|---|
| `demo_mode` | `no_tags` | Format used for demonstrations |
| `ref_model_mixup_alpha` | `0.05` | Reference-model synchronization mix |
| `generation_prompts_per_device` | `32` | Generation batch contribution per process |
| `max_completion_length` | `16384` | Maximum generated completion length |
| `vllm_gpu_memory_utilization` | `0.3` | vLLM memory fraction |

```bash
bash scripts/train_sdft.sh science
```

SDFT has the largest memory footprint because it loads student and reference
models and performs online generation.

## OPLoRA

OPLoRA freezes the base linear layer and trains a low-rank adapter. Before the
adapter contribution is applied, its input and output are projected away from
the selected principal subspaces. The adapter is merged into the saved model at
the end of training.

| Setting | Default |
|---|---:|
| LoRA rank / alpha | `32 / 32` |
| Dropout | `0.05` |
| Projection rank | `128` |

```bash
bash scripts/train_oplora.sh science
```

## Sculpting

Sculpting first estimates a retention ratio for every transformer block from
hidden-state similarity on the training set. It then installs gradient hooks
that remove the protected two-sided component for the selected matrices.

| Setting | Default |
|---|---:|
| Minimum retention ratio | `0.1` |
| Target retention ratio | `0.8` |
| Calibration samples | all examples |

```bash
bash scripts/train_sculpting.sh science
```

For exact values and target-module lists, use the files in
[`configs/method/`](../../configs/method) as the source of truth.

