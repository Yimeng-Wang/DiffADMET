# Optional user-owned CSV input contract

The default public reproduction inputs are BIN, described in [BIN_DATA_FORMAT.md](BIN_DATA_FORMAT.md). This page applies only to user-owned CSV data; the original study CSVs are not distributed.

The authoritative task order is `configs/task_schema.json`: 108 classification tasks followed by 27 regression tasks. `SMILES` may be first or last, but the relative order of all task columns must match the schema exactly. Task names are case/space sensitive, including existing leading or repeated spaces. Do not strip or rename them silently. Extra metadata columns, duplicated headers, infinities, non-numeric labels, and non-binary observed classification labels are rejected.

Each row describes one molecule observation. Repeated SMILES within a split retain separate label rows; graph geometry may be cached by SMILES. Missing labels remain masked and explicit zero remains observed. Regression values must use consistent units/transformations for each endpoint; these cannot be inferred from an endpoint name.

## Split filenames

Supply the already assigned folds using the filenames below. The training entry point reads these files directly; it never regenerates or reassigns folds. A split-generation helper is not included in this release.

| Filename under `--data-dir` | Required |
| --- | --- |
| `0admet135_train3.csv` … `4admet135_train3.csv` | selected folds |
| `0admet135_valid3.csv` … `4admet135_valid3.csv` | selected folds |
| `admet135_test3.csv` | optional fixed test set |
| `admet135_cancer.csv` | optional external evaluation set |

The full original `admet135_1.csv` is no longer required by the training entry point. Its former header lookup is replaced with the public schema. Existing `.pkl` normalizers are also unnecessary; fit statistics are recalculated from each training split, used in training/evaluation and saved as the local `normalizer.json`.

Training reads the named existing split files directly and preserves their assigned records. No folds are regenerated.

## Task relationship matrix

Pass `--relation-matrix` pointing to your own local CSV with 135 rows and 135 task-named columns, in schema order. Values must be binary 0/1, with at least one allowed edge in every row and column. An optional first row-label column is accepted only when it matches the task names exactly. The numeric rows without such a label column must already be in schema order.

The implementation computes `transferred[i] = sum_j W[i,j] * task_features[j]`; the original matrix mode normalizes weights along column dimension 0. Thus matrix entry `[i,j]` contributes source task `j` to target task `i`. Preserve that convention when constructing an input matrix. Diagonal self-relations are normally needed. BIN mode includes the original matrix in the separate authorized dataset; its historical selection rule is not independently reconstructed. CSV mode requires the user's own supplied matrix. No statistical relation estimation is claimed or invented by this release.

## Local outputs

`outputs/<mode>/fold_<fold>/` stores `best.pth`, `normalizer.json`, `run_manifest.json`, `history.json`, and endpoint-metric CSVs: `<fold><split>_evalue_c.csv` or `<fold><split>_evalue_r_denorm.csv`. The best checkpoint is saved, reloaded for evaluation and retained. No per-record probability/prediction/label/mask CSVs are generated. CSV mode may create graph binaries under `cache/`; these local-input caches remain excluded from upload. Public BIN mode reads the supplied graphs directly.

Normalization uses population standard deviation (`ddof=0`) on observed training regression labels. Zero standard deviation falls back to 1; a training task with no observations uses mean 0 and standard deviation 1. Such a task is not scientifically trained and should not be interpreted as a validated endpoint.
