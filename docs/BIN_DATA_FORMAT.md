# Binary reproduction data format, version 1

All paths below are relative to `data/reproduction/`. The original CSVs and SMILES mapping are unnecessary for the reader.

## Two binary encodings

The `.bin` suffix identifies dataset artifacts, not one universal serialization format:

| File | Encoding | Contents |
| --- | --- | --- |
| `graphs/graphs_NNNN.bin` | DGL native `save_graphs` | Graph inputs and integer `molecule_ids` |
| `graph_index.bin` | NumPy `savez_compressed` | Integer `shard` and `position` arrays indexed by molecule ID |
| `splits/fold_N_train.bin`, `fold_N_valid.bin`, `test.bin`, `external.bin` | NumPy `savez_compressed` | `labels`, `masks`, `molecule_ids`, `record_ids` |
| `relations.bin` | NumPy `savez_compressed` | Binary task relationship matrix |
| `normalizers/fold_N.json` | JSON | Original 27 regression means/stds |

Read numeric BIN with `np.load(path, allow_pickle=False)`; read graph BIN with `dgl.load_graphs`. The exported dataset does not require loading legacy Python pickle objects.

## Graph correspondence

Within a graph shard containing N molecules, the first N graphs are 2D atom/bond graphs and the next N are the corresponding 3D bond/angle graphs. Metadata `molecule_ids[i]` identifies graph `i` and graph `N+i`. `graph_index.bin` maps that molecule ID to its shard and local position.

Only model input fields are exported: node `hv` and edge `he`. Dimensions remain 78/13 for 2D and 4/1 for 3D. Topology, node ordering, edge ordering and tensors are copied from the existing cache. No conformer is regenerated. A missing 3D edge feature on a graph with zero edges is represented by an empty `(0,1)` tensor and recorded as a repair; nonempty missing features are rejected.

## Labels and masks

Each profile's label array is float32 of shape `(records,135)`, matching the original model-input conversion. The first108 tasks are classification, the last27 regression, in `configs/task_schema.json` order. Masks are uint8 0/1 of the same shape and become float32 for training. Original blank/NaN values are stored as label0 with mask0; an explicitly observed zero remains label0 with mask1.

`record_ids` are unique across exported profiles and do not encode SMILES or original molecule identifiers. The data loader uses these as internal record identifiers; current evaluation exports only endpoint metrics, with no per-record ID table. `molecule_ids` identify the cached graph; the same ID can appear in different original splits/folds. Its graph remains structurally informative. No original-to-release mapping is distributed.

## Original dataset handling

Labels are constructed from each original CSV split, retaining its first occurrence for each exact SMILES, as the supplied legacy loader did. The final valid cached graph occurrence is used if a molecule appears in multiple cache batches. The complete per-profile row ledger is in the manifest. The current last-written label-cache BIN is not treated as a complete data source; it can correspond to one recently loaded split only.

Original fold normalization values are copied from the provided artifacts and are the default for BIN training. The manifest compares them with a fresh fit on raw training rows. `--normalizer-policy fit` instead refits on effective exported rows or a selected diagnostic subset; record this change when interpreting results.

The original task relation matrix and original splits are preserved. Training directly reads the provided profiles. The manifest retains the historical split-audit statistics as dataset provenance.

## Verification

Run `scripts/validate_binary_dataset.py --verify-hashes --deep`. The manifest contains the approved dataset file list and SHA-256 hashes. Fields are numerical; graph metadata contains only integer IDs. These checks establish file integrity and schema consistency, not biological accuracy or a confidentiality guarantee.


### Molecule overlap versus endpoint-observation overlap

The supplied folds share exact-source molecule identities between training, validation and test. An independent check of **all five folds** found zero shared observed `(molecule_id, endpoint)` pairs between any two of those sets: shared molecules have measurements at different endpoints. This is a distinction between molecule-disjoint evaluation and endpoint-observation holdout; it is not a claim that entirely unseen-molecule generalization has been established. The identity check follows the original exact SMILES keys and does not audit all canonical aliases. Both molecule counts and endpoint-observation counts are preserved in the manifest. The external set shares no such molecule IDs with any fold's training set.

## Preset five-fold assignments

For N = 0–4, `splits/fold_N_train.bin` comes from the supplied `Nadmet135_train3.csv` and `splits/fold_N_valid.bin` comes from `Nadmet135_valid3.csv`. All ten files were independently verified to retain exact sample membership, row order, labels and masks. No new K-fold partition is created during conversion or training. The fixed test profile also remains as supplied. Anonymous molecule IDs only replace the private identity representation.

Current training consumes the predefined train/validation profiles and evaluates the fixed test and external profiles when present. External data use the fold's normalization parameters and are excluded from optimization, validation monitoring and checkpoint selection. Existing archive inventories are unchanged.
