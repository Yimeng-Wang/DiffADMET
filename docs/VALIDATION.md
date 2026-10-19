# Executed release validation

The current installation definition is `environment.yml`, supplied from the Linux Python3.8/CUDA11.3 training environment. The checks below describe the historical local reference environment and do not certify a fresh Linux environment installation.

Date: 2026-10-11. Environment: Windows, Python 3.11.7, PyTorch 2.2.2+cpu, DGL 1.1.2, DGL-LifeSci 0.3.2, RDKit 2023.9.6. GPU execution, full 1,000-epoch training and historical paper/website-checkpoint equivalence were not tested.

## Real processed data

- 180,100 graph-input identities, represented by 364 graph shards containing both 2D and 3D graphs. Approximately 3.38 GB of approved files before outer ZIP compression.
- 12 label/mask profiles: five training splits, five validation splits, one fixed test split and one external split. They total 829,848 profile records because molecules/observations are reused across cross-validation folds; this is not the number of distinct compounds.
- All source split rows are retained in this release. The legacy first-occurrence rule was applied, but these provided per-split CSVs have no within-split exact-SMILES duplicates to remove.
- Coverage audit confirmed that the existing cache covers all 180,100 input identities. No conformers were regenerated and no empty-edge feature repair was necessary.
- Every exported graph round-tripped with exactly equal topology, node/edge ordering, and `hv`/`he` tensors. Exported native graph fields contain only model inputs and integer anonymous IDs.
- Independent source projection compared every public label and mask against the original CSV conversion, and every anonymous graph ID against the private original cache index: all matched exactly. Explicit source zeros were preserved as observed and source blanks as masked.
- The original 135 x 135 task relation matrix was copied exactly. All five original normalization artifacts were copied exactly into JSON. Comparisons with fresh raw-training-row fits differed by at most 1.78e-15 in means and 4.45e-16 in standard deviations.
- Full independent validation of all dataset hashes, graph shards, feature dimensions, profile IDs/labels/masks, normalization and actual file membership passed.
- Data fields contain no plaintext SMILES, no original identity map, no CSV/pickle containers and no prediction-model weights. Numerical graph inputs retain molecular information, and real graph-label pairing is publicly authorized; no anonymity guarantee is made.

## Split audit

The original fixed splits were preserved. Train/validation molecule overlaps range from 1,400 to 1,419; train/test overlaps from 880 to 926; validation/test overlaps from 224 to 262. All five folds have **zero shared observed molecule/endpoint labels** between train, validation and test under the original exact identity keys. Thus shared molecules carry measurements at different endpoints. No canonical-alias audit or unseen-molecule performance claim is made. The external set has no molecule-ID overlap with any training fold. Details are in `configs/reproduction_manifest.json`.

## Training and reader checks

- Before the metrics-only cleanup, anonymous-ID BIN smoke tests passed both classification and regression: one artificial-data epoch, saved-normalizer loading, scheduler/validation, new checkpoint save/reload, and train/validation/test evaluation with `sample_id` output columns.
- Actual real BIN inputs passed the validation-only training CLI for all five folds, preserving legacy molecule overlaps explicitly.
- Before the metrics-only cleanup, real-data diagnostic runs passed classification and regression on fold0 with deterministic subsets of up to128 records per split, one epoch, batch32 and CPU execution. Both saved/reloaded newly trained weights and evaluated train/validation/test/external splits. Their weights and outputs remain outside the publication directory.
- Random/unsorted graph selection was checked against full-shard loading, preserving the index-to-graph association.
- Previously executed canonical-feature, seed-aligned geometry, missing-label normalization, original difficulty-aware loss and repaired training-architecture checks passed. Later fixed-path cleanup removes inactive branches; its shared-weight equivalence and metrics-only output checks are described below.
- All public Python source compiles; Ruff F checks pass.

## Packaging

The publication builder separately verifies an explicit source allowlist and an approved real-data inventory. It checks every ZIP member and SHA-256 hash before delivery. Raw CSVs, private mapping, legacy pickle caches, pretrained weights, validation outputs and website service code are excluded. Prepared assets remain local until uploaded by the owner. No fabricated external dataset URL or DOI is supplied.

## Preset-fold recheck (2026-10-11)

All ten original training/validation CSVs were independently compared with the released BIN records using the private source-to-ID mapping outside the publication directory. Every sample membership and original row order matched, as did every label and observation mask. The five-fold files were not regenerated or changed.

| Original fold | Training records | Validation records |
| --- | ---: | ---: |
| 0 | 129,019 | 32,802 |
| 1 | 129,000 | 32,802 |
| 2 | 129,006 | 32,806 |
| 3 | 129,019 | 32,788 |
| 4 | 129,002 | 32,801 |

The optional split-generation script and its usage instructions were removed from the source release to avoid accidental repartitioning. The real BIN files and data ZIP parts remain unchanged.

## Intermediate fixed dual-graph and metrics-only cleanup (2026-10-11)

- Original active training-entry values were rechecked: 1,000 epochs, batch2,048, seed66, classification learning rate0.0003/patience50, regression learning rate0.0006/patience40. The data-loader worker default is8, matching the original normalized loaders. Runtime CPU checks explicitly used0 workers.
- Model forward accepts only the pair of graph/feature triples and returns logits. All six learner calls (train, validation, evaluation for each objective) supply no architecture keywords. Alternative fusion/relation modes and disabled cross-attention/alignment/adversarial branches are absent.
- Resetting seed66 and constructing the new model twice produced exactly equal state tensors.
- After loading the same effective reference weights, new versus previous fixed-path logits had maximum absolute difference0. Active gradients and classification/regression difficulty-aware loss components matched within floating-point tolerance.
- Classification `_evalue_c.csv` and regression `_evalue_r_denorm.csv` matched the corresponding previous evaluator tables under those shared weights, including missing-label handling and endpoint order.
- Before restoration of the four training artifacts, artificial BIN classification/regression one-epoch training, early stopping, temporary best-checkpoint reload and evaluation passed. Only the required per-split metric CSVs remained; no persistent checkpoint, output normalizer, history, run manifest or per-record prediction/label/mask file was produced.
- Before restoration of the four training artifacts, real BIN fold0 checks also passed for both objectives with up to128 records per split, one epoch, batch32, CPU and0 workers. Each mode produced exactly four metric CSVs (train, valid, test, external), with108 classification or27 regression endpoint rows in the original schema order.
- The complete five preset folds passed validation-only execution with their original counts. All three original real-data ZIP parts and their inventory remained byte-for-byte unchanged.

These checks verify the retained mathematical path and output contract. Removing inactive modules changes initialization/dropout RNG consumption; they do not establish identical historical from-scratch trajectories or complete1,000-epoch paper-score equivalence. GPU execution and eight-worker server execution were not tested here.

## Restored training artifacts (2026-10-11)

At the owner's request, each training fold again retains `best.pth`, `normalizer.json`, `run_manifest.json` and `history.json`. CSV output remains limited to classification `_evalue_c.csv` or regression `_evalue_r_denorm.csv`; no per-record probability/prediction/label/mask files are produced.

- Both objectives passed one-epoch artificial BIN training and best-checkpoint reload. The result-directory check required exactly the four training artifacts plus the expected per-split metric tables.
- Normalization JSON contained27 means/stds and a fitted state. The run manifest recorded fixed seed66, actual epoch/batch overrides, fold selection and code/input hashes. History contained one epoch and135-element train/validation task-loss arrays.
- Hash comparisons confirmed that the fixed model, both learners/evaluators, input readers and all training/task/data configurations were unchanged by this artifact restoration. Original default settings and preset five-fold membership remain in effect.
- All three real-data ZIP parts still matched their recorded byte sizes and SHA-256 hashes. No data re-export or repartition was performed.

The current release retains these four training artifacts; the intermediate temporary-checkpoint-only behavior documented above is superseded. They are local run outputs excluded from source/data publication archives. GPU/full-length training was not repeated for this output-only change.

## Direct preset-fold loading (2026-10-11)

The training entry now reads the supplied folds directly. The former runtime cross-split molecule guard, its command-line switch and warnings, and the per-run summary field were removed. Historical split-audit statistics remain in the unchanged dataset manifest.

- The complete real BIN five-fold validation passed without an override parameter or runtime warning. All original train/validation/test/external row counts were preserved.
- Classification and regression artificial-data training passed, retaining `best.pth`, `normalizer.json`, `run_manifest.json`, `history.json` and the required per-split endpoint-metric CSVs.
- CLI help and source/reference scans confirmed that the former parameter/check helpers were absent. Python compilation and Ruff F checks passed.
- Hash comparisons confirmed the model, both learner/evaluators, normalizer, original default configurations, schema and data inventory remained unchanged. All three real-data ZIP parts still matched their recorded hashes.

This change affects input-guard behavior and runtime metadata only; it does not regenerate folds, change graph/label inputs or alter the training objective.

## Runtime provenance without dependency versions (2026-10-11)

At the owner's request, dependency-version collection was removed entirely from the training entry and run manifest. Actual settings, selected folds and source/input hashes remain recorded; `best.pth`, `normalizer.json`, `run_manifest.json`, `history.json` and the required endpoint-metric CSVs remain outputs.

- Classification/regression artificial-data one-epoch training, best-checkpoint reload and evaluation passed. Tests verified the four training artifacts and absence of a dependency-version field.
- Source scans confirmed the version-reader function, module-version lookups and distribution-metadata queries were absent. Static checks passed.
- Hash comparisons confirmed the model, both learners/evaluators, data readers, original defaults, task/data inventories and all three real-data ZIP parts were unchanged.

The change removes optional environment-version logging only. It does not alter training, preset split membership or the retained output artifacts. Tests ran in the previously documented Python3.11 CPU reference environment.

## Conda environment and documentation update (2026-10-11)

The owner-provided `environment.yml` now defines installation, replacing the separate dependency list. Its environment name, channels and package/version specifications are preserved; only the author-machine installation prefix was removed. The current specification is Linux Python3.8.20, PyTorch1.12.1, CUDA toolkit11.3.1 and dgl-cu1130.9.1.post1. English/Chinese READMEs now cover environment creation, data extraction, the original five-fold training commands and retained output files.

- Parsed environment/CI YAML successfully; confirmed environment-name activation and direct use of the environment file in CI. The existing artificial-data smoke test remains configured.
- Verified that all training/model/data-reader sources and all training/task/data configurations retained their previous hashes. All public Python sources parsed using Python3.8 grammar, and Ruff F checks passed with the py38 target.
- The three real-data ZIP parts retained their recorded sizes and hashes. Source publication membership and regenerated source-archive hashes were checked separately.

This documentation/package change did not rerun model training. A fresh solve/install of the supplied Linux Conda environment and execution of the modified GitHub Actions workflow were not performed on this Windows host; the historical local runtime checks above remain identified by their actual environment.

## Main-split evaluation and compact reporting (2026-10-11)

Current training reads and evaluates only train, validation and test. The additional evaluation-set read/evaluation/output paths were removed from BIN/CSV training. Existing data ZIPs and inventories remain unchanged for compatibility; unused cached profiles are not read or evaluated by the training entry.

- Regression tests inserted an unreadable unused profile into a legacy-style manifest. Both classification and regression completed training without accessing it and produced exactly three metric tables plus `best.pth`, `normalizer.json`, `run_manifest.json` and `history.json`.
- The old fixed per-endpoint report lists and grouped console output were removed. Tiers now use each split's current classification AUC or regression `R2_denorm`: Robust >0.9, Intermediate0.8–0.9 inclusive, Difficult <0.8. Undefined/nonfinite metrics are excluded. Exact boundary, empty-valid-set and missing-value tests passed.
- Training emits one progress summary initially, every ten epochs and at termination. Evaluation emits one summary per main split. Background graph-loading progress bars and early-stop counter chatter are suppressed; full endpoint values remain in the CSVs. Explicit zero precision for no positive predictions keeps the previous default numeric result without repeated warnings.
- Before/after metric tests and shared-model evaluator comparisons found unchanged CSV values. AST comparisons confirmed the original difficulty arrays and difficulty-weight/loss methods were unchanged; performance-tier labels do not affect training weights.
- Real BIN fold0 classification/regression CPU checks passed with one epoch, batch32 and up to128 samples per main split. Each mode produced exactly the four training artifacts and three correctly ordered metric tables, with no extra evaluation outputs. Full validation-only reads passed for all five original folds and preserved their original main-split row counts.
- Python3.8 grammar checks, static checks and publication checks passed. Model, default configuration, environment and original data inventory hashes were unchanged, as were all three data ZIP parts.

These checks ran in the historical local CPU reference environment. Full-length GPU training was not repeated. The global seed remains set once before the selected folds; removing an evaluation loader can change subsequent DataLoader RNG consumption, so identical later-fold historical trajectories are not claimed.

## Mean metrics within performance tiers (2026-10-12)

Evaluation summaries now report the overall endpoint mean and the arithmetic mean within each current performance tier, replacing tier counts and the valid-endpoint count. Classification uses AUC; regression uses the existing `R2_denorm`. The thresholds remain Robust >0.9, Intermediate0.8–0.9 inclusive and Difficult <0.8. Nonfinite values are excluded from tier means; an empty tier displays `nan`.

- Exact-boundary, arithmetic-mean, missing/nonfinite-value and empty-tier checks passed.
- AST comparisons confirmed that both learners differ only in their print statements, and all metric calculations in `Meter` are unchanged. Model and configuration hashes also remain unchanged.
- The existing artificial BIN smoke test passed for both objectives: one training epoch, checkpoint reload, three mean-metric summary lines and exactly the four retained training artifacts plus three endpoint-metric CSVs.
- Python3.8 syntax, release checks and Ruff F checks passed. Only the source publication archive was rebuilt; original BIN data and five-fold assignments were not modified.

Runtime checks used the previously documented local Python3.11 CPU reference environment. Full-length GPU training was not repeated for this logging change.

## Restored external evaluation (2026-10-12)

External evaluation is restored at the owner's request. BIN mode uses the optional `external` profile; CSV mode uses optional `admet135_cancer.csv`. After checkpoint reload, each fold evaluates this set using the same fold normalizer and saves `<fold>external_evalue_c.csv` or `<fold>external_evalue_r_denorm.csv`. The one-line overall and per-tier mean metrics remain in effect. External labels are excluded from optimization, normalization fitting, validation monitoring and checkpoint selection. This supersedes the earlier main-split-only entry behavior.

- The artificial BIN smoke test passed for classification and regression, including external logs, external input hashes and exactly four training artifacts plus four endpoint-metric CSVs per fold.
- All five original real BIN folds passed validation-only loading, including290 external records per fold. CSV input validation also passed using artificial inputs and the restored optional external filename.
- The original `admet135_correlation_bool_choose2.csv` was compared directly with the published `relations.bin`; all135×135 entries match exactly. Default BIN training passes this matrix to the model's task-transfer mechanism.
- Hash comparisons confirm that the model, learners, metric calculations/mean summaries, default configurations, environment, task order, normalization implementation and original data inventories remain unchanged. The source archive was rebuilt; existing data ZIPs were not modified.

Runtime checks used the previously documented local Python3.11 CPU reference environment. Full-length GPU training was not repeated. Seed66 and original five-fold membership remain unchanged; restoring evaluation iteration may change RNG consumption before later folds and is not a claim of identical historical trajectories.

## Fixed endpoint difficulty reporting (2026-10-13)

The previous reporting helper incorrectly reassigned groups using each split's current model scores. It is replaced with fixed endpoint labels from the figure source `Type.csv`, column `Type1`, stored in the source schema without molecular records. Mapping: >0.9→Robust,0.8–0.9→Intermediate,0.7–0.8/<0.7→Difficult and new→Novel. These are fixed source labels; current training/test scores do not determine group membership.

- All135 task names match the source table. Classification groups contain32 Robust,48 Intermediate,17 Difficult and11 Novel endpoints; regression contains3 Robust,5 Intermediate,10 Difficult and9 Novel endpoints. Original task strings and order are preserved.
- Tests assign every endpoint a score of0.99 and confirm that all four fixed groups still report0.99. Changing scores or reversing task order does not change group membership. Missing/nonfinite values are excluded only when calculating group means; a fixed group without a valid metric displays `nan`.
- Classification and regression artificial BIN one-epoch training/evaluation passed for train, validation, test and external, with nonempty Difficult summaries and the unchanged four training artifacts plus four endpoint-metric tables.
- AST comparisons confirm that the learners differ only in reporting print statements and the metric implementation is unchanged. Hash checks confirm the training entry, model, default parameters, normalization, original task order, environment and data inventories remain unchanged.
- Python3.8 syntax, Ruff F and source-release checks passed; the source archive was rebuilt. Existing BIN data and predefined fold assignments were not regenerated.

Checks used the previously documented local Python3.11 CPU reference environment. Full-length GPU training was not repeated for this reporting correction. Previous dynamic score-tier validation notes above are superseded.

## Concise READMEs and public Type1 table (2026-10-14)

English/Chinese READMEs and the owner's attached README now omit the detailed optimizer/default-training table and random-seed instructions. Installation, five-fold training commands and output descriptions remain.

The previously referenced `Type.csv` was an author-side figure input absent from the source release. `configs/Type.csv` is now included with exactly `Task` and `Type1`. All135 rows preserve the source's endpoint names, Type1 values and row order; Type2/Type3 are excluded. The original author-side table is preserved. Every row maps to the existing fixed-tier dictionary, with no change to endpoint assignments or ordering.

The source allowlist and Git ignore exception include only this endpoint-category CSV. All training/model/metric sources, original optimizer configurations, environment and real-data inventories retain their previous hashes. CSV visual/structural checks, Python syntax and static publication checks passed. No model training was repeated for this documentation/table-publication change; only the source ZIP was rebuilt.

## English-only publication (2026-10-14)

At the owner's request, the additional translated README and its links were removed from the source publication. Three remaining runtime error/warning strings were translated to English. A local run log is preserved outside the publication directory. The publication checker now rejects CJK characters in source filenames and text.

- The three edited Python modules retain identical ASTs after normalizing the message translations; training, Huber thresholds, evaluation values, fixed endpoint groups and output behavior are unchanged.
- All other training/model/data-reader sources, default configurations, task metadata, environment and data inventories retain their previous hashes. Existing data ZIPs were not rewritten.
- Source publication checks, Python3.8 grammar and Ruff F checks passed. All publication text files and text/name entries in the source and three data archives passed the CJK scan. The source archive was rebuilt without the removed documentation file or runtime log.

No model training was repeated for these language-only edits.
