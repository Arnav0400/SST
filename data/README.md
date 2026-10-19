# Datasets

This directory contains the saved Hugging Face datasets used by the default SST
configs.

| Dataset | Examples | Arrow bytes | SHA-256 |
|---|---:|---:|---|
| Science | 9,000 | 81,992,912 | `274027cd568b93be5f93c3d202075e02dc503248469dd95296cb09d8316aee96` |
| Medicine | 8,996 | 27,345,448 | `b9379e95452df1abd13e30d569754c2ba3d1fcd3564f0fdc632e712902dced25` |
| Pathology | 8,999 | 25,425,040 | `a1ce9795ba6cd0849ea39b5331010daeb2cfe69ac85eacdfd9b935cd2d028d46` |
| FinQA | 5,000 | 25,351,600 | `b3aa0951a60e60adccdd6d8d29d8cb406f2da09b8a854f2df53373dccec86e78` |
| **Total** | **31,995** | **160,115,000 (153 MiB)** | |

The Arrow payloads are stored with Git LFS:

```bash
git lfs pull
python scripts/verify_data.py
```

Generated `cache-*.arrow` files and intermediate JSONL files are not tracked.

## Provenance

The bundled snapshots are derived from the paper's training sets:

| Snapshot | Upstream source | Upstream terms |
|---|---|---|
| Science | [Open R1 Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts), science subset | The source subset comes from NVIDIA's Llama-Nemotron post-training data; consult its per-sample terms |
| FinQA | [FinQA](https://github.com/czyssrs/FinQA) and [ConvFinQA](https://github.com/czyssrs/ConvFinQA) | MIT |
| Medicine, Pathology | [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) questions with generated reasoning traces | Apache-2.0 metadata for MedMCQA; traces are additional generated annotations |

These links are provided for attribution and provenance. Review the upstream
terms for the exact samples and intended use before redistributing another
mirror.
