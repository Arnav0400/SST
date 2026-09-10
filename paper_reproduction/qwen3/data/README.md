# Qwen3 reproduction datasets

This directory contains the exact saved Hugging Face dataset snapshots used by
the main Qwen3 experiments. Training reads these files directly; it does not
depend on the repository's old production data tree or data-building scripts.

| Dataset | Examples | Arrow bytes | SHA-256 |
|---|---:|---:|---|
| Science | 9,000 | 81,992,912 | `274027cd568b93be5f93c3d202075e02dc503248469dd95296cb09d8316aee96` |
| Medicine | 8,996 | 27,345,448 | `b9379e95452df1abd13e30d569754c2ba3d1fcd3564f0fdc632e712902dced25` |
| Pathology | 8,999 | 25,425,040 | `a1ce9795ba6cd0849ea39b5331010daeb2cfe69ac85eacdfd9b935cd2d028d46` |
| FinQA | 5,000 | 25,351,600 | `b3aa0951a60e60adccdd6d8d29d8cb406f2da09b8a854f2df53373dccec86e78` |
| **Total** | **31,995** | **160,115,000 (153 MiB)** | |

Only the canonical `data-00000-of-00001.arrow`, `dataset_info.json`, and
`state.json` files are included. Generated `cache-*.arrow` files and the
medicine JSONL intermediates were intentionally omitted.

The Arrow payloads are stored with Git LFS:

```bash
git lfs install
git lfs pull
python paper_reproduction/qwen3/verify_data.py
```

The source snapshots' `dataset_info.json` files do not declare licenses. Confirm
the redistribution terms of the underlying source datasets before publishing a
public mirror or derived archive.
