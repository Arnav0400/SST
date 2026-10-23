# SST as an optimizer wrapper

SST takes an optimizer, calls that optimizer's `step()`, then projects the
resulting parameter update outside fixed principal left and right weight
subspaces.

The inner optimizer receives the original gradients. Its momentum, variance,
weight decay, and parameter groups are not rewritten by SST. The fixed singular
triples `(U, s, V)` define the affine projection

```text
W <- U diag(s) V^H + (I - U U^H) W (I - V V^H).
```

## Independent PyTorch implementation

`sst_torch.optimizer.SST` depends only on PyTorch; it does not import Transformers
or Accelerate for ordinary training. No optimizer-specific subclass is needed:

```python
import torch
from sst_torch import SST

inner_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = SST(inner_optimizer, model=model, rank_fraction=0.99)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Pass module suffixes or full-match regexes in `target_modules`, or `"all-linear"`
for linear weights. Without a filter, all owned trainable matrices with both
dimensions at least two are selected. Supplying `model` is optional for ordinary
parameters, but required for name filtering and FSDP1/DeepSpeed parameter access.

Schedulers use the wrapper's shared parameter groups normally. Closures and
optimizer step/state-dictionary hooks are retained. AMP overflow and Accelerate
gradient-accumulation skips do not project.

## Native Transformers implementation

The Transformers contribution contains its own implementation in
`src/transformers/optimization_sst.py`, publicly exported through
`transformers.optimization` and `from transformers import SST`. It does not
depend on this repository or an `sst-torch` installation.

The draft is [Transformers PR #49892](https://github.com/huggingface/transformers/pull/49892).
Until it is merged, install the development branch:

```bash
pip install "transformers @ git+https://github.com/Arnav0400/transformers.git@codex/sst-optimizer-integration"
pip install "accelerate>=1.15.0"
```

Use the **native** class with the new Trainer entry point:

```python
from transformers import SST, Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="sst-output", optim="adamw_torch"),
    train_dataset=train_dataset,
    optimizer_wrapper=(
        SST,
        {"rank_fraction": 0.99, "target_modules": [r".*self_attn.*", r".*mlp.*"]},
    ),
)
trainer.train()
trainer.train(resume_from_checkpoint="sst-output/checkpoint-100")
```

`optimizer_cls_and_kwargs` can select another inner optimizer. SST is applied
after Accelerate prepares the model/optimizer and before scheduler construction.
This matters because FSDP changes parameter layouts, and DeepSpeed performs its
optimizer update inside its engine. Do not wrap an unprepared DeepSpeed
optimizer and bypass the engine's step boundary.

The standalone and native classes share the projection implementation, but
Trainer's engine-aware checkpointing recognizes its native class. Use
`transformers.SST` for Trainer, and `sst_torch.SST` for standalone training loops.

### Alignment with merged Transformers contributions

The class-plus-keyword-arguments API follows the custom optimizer entry point
merged in [#34358](https://github.com/huggingface/transformers/pull/34358).
Keep `optimizer_cls_and_kwargs` responsible for constructing the inner optimizer;
`optimizer_wrapper` composes SST around it after distributed preparation.
This avoids a separate optimizer name for each SST/base-optimizer combination
and does not require a Trainer subclass.

The [GaLore integration (#29588)](https://github.com/huggingface/transformers/pull/29588)
and [APOLLO integration (#36062)](https://github.com/huggingface/transformers/pull/36062)
provide precedents for target-module selection, discoverable optimizer examples,
and Trainer regression tests. They project gradients for memory-efficient
optimization; SST instead projects the applied update while preserving the
inner optimizer's state. Their merge does not imply approval of SST.

## Distributed parameter handling

| Backend | How SST accesses and writes the update |
|---|---|
| Single device / DDP | Projects replicated optimizer parameters in place. |
| FSDP1 | Gathers one wrapping unit at a time with `summon_full_params`, then writes back local shards. Supports original and flat parameters. |
| FSDP2 / DTensor | Assembles one selected global matrix, projects it, and redistributes to its original mesh and placements. |
| DeepSpeed ZeRO-1/2 | Projects the full fp32 master parameter and updates the low-precision model copy. |
| DeepSpeed ZeRO-3 | Projects fp32 master partitions and writes the rounded result back through gathered model parameters. |

All participating ranks must call the wrapper. In a custom Accelerate loop,
attach SST **after** `accelerator.prepare`. Trainer handles this ordering.

The inner optimizer must support the backend and parameter layout itself.
For example, two-dimensional-only Muon cannot optimize FSDP1's flat
one-dimensional parameters. SST does not change an optimizer's distributed
algorithm or make unsupported parameter layouts supported.
Transformers' `Adafactor` works with DDP, but its factored buffers and RMS
checkpoint metadata are not FSDP-compatible. Use `torch.optim.Adafactor` for
the tested FSDP paths.

For FSDP2 CPU offload with gradient clipping, configure a CPU-capable collective
backend, for example `TrainingArguments(ddp_backend="cpu:gloo,cuda:nccl", ...)`.
NCCL alone cannot reduce CPU DTensor gradients. SST temporarily gathers offloaded
matrices on their mesh device and writes the result back to their CPU shards.

Exact projection requires fp32/fp64 optimizer parameters or fp32 master weights.
Autocast bf16/fp16 compute with fp32 optimizer parameters is supported. Direct
bf16/fp16 parameter writeback is approximate. XLA, SageMaker model parallelism,
and differentiable optimizer steps are not supported.

Initialization performs an SVD per target. Bases are kept on CPU, and distributed
steps require gathers. These costs are significant for large models; the
correctness tests are not throughput or large-model benchmarks.

## Checkpoint ownership

An ordinary `optimizer.state_dict()` includes all inner state plus the fixed
basis under `"sst"`. To restore a custom loop without a fresh SVD:

```python
optimizer = SST(inner_optimizer, model=model, initialize=False)
optimizer.load_state_dict(saved_optimizer_state)
```

Trainer saves the original basis automatically. FSDP and DeepSpeed retain their
native optimizer checkpoints and additionally save `sst_state.pt`, because
engine checkpoint conversion does not retain arbitrary wrapper keys.
Subgroup/submesh-owned targets instead use `sst_state_rank_<rank>.pt`, with
rank and mesh topology recorded and checked on resume. Standalone loops should
likewise save each rank's optimizer state when `optimizer.requires_per_rank_state`
is true. These rank-specific checkpoints require the same topology on resume.

The FSDP1 sharded restore path allocates saved optimizer buffers from checkpoint
metadata. It does not take a dummy optimizer step to create state. Missing or
mismatched bases fail explicitly; they are never rebuilt from trained weights.
The rank and basis in the checkpoint take precedence when resuming.

FSDP sharded training checkpoints are distinct from a consolidated model export.
Before `trainer.save_model()` for ordinary `from_pretrained()` inference, request
`FULL_STATE_DICT` on the FSDP plugin.

## Validation

The upstream branch contains optimizer unit tests, Trainer tests, and offline
distributed end-to-end tests:

```bash
python -m pytest tests/optimization/test_sst.py tests/trainer/test_trainer_sst.py
RUN_SLOW=1 python -m pytest tests/trainer/distributed/test_trainer_distributed_sst.py
```

The distributed script uses a small randomly initialized Llama and synthetic
tokens, without model downloads. It checks training, evaluation, saved-model
generation, unchanged inner optimizer state during projection, protected
subspaces, replica equality, and fixed-basis checkpoint resume without SVD.
An additional four-GPU DTensor test checks separate two-rank submeshes and
rejects another rank's fixed-basis checkpoint.
The PR records the actual tested optimizer/backend/precision configurations.

The standalone repository's tests run with:

```bash
python -m unittest discover -s tests
```

## License and contribution status

SST's original standalone code is MIT licensed. Vendored SDFT/TRL components
retain their Apache-2.0 headers and notices. The Transformers contribution follows
that repository's Apache-2.0 license.

The upstream PR remains a draft for human review and maintainer coordination.
It is AI-assisted; no non-AI-authorship attestation is claimed. Transformers'
contribution policy still requires maintainer approval and applicable contributor
review before this can be made ready or merged.
