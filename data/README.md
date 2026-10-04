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
The saved metadata does not declare source licenses; confirm source
redistribution terms before publishing a public data mirror.
