# Data availability

The project owner has authorized public reproduction using real molecular graphs and their corresponding labels **without plaintext SMILES**. This supersedes the earlier source-only/controlled-data preparation. The real binary data package is prepared separately from the GitHub source ZIP.

## Included

- Cached 2D atom/bond and 3D bond/angle graph inputs.
- Real classification/regression labels and explicit missing-label masks for the original five training/validation folds, fixed test set and external evaluation set used by the current training entry.
- Anonymous molecule IDs linking graphs to records, and anonymous record IDs used internally by the data loader.
- Original task relationships and original fold-specific normalization values converted to JSON.
- File hashes, schema information, source/effective row counts and original split-overlap counts.

## Excluded

Original modeling CSVs, plaintext SMILES identifiers, the private original-to-release ID mapping, legacy cache/pickle containers, pretrained prediction weights, and website service code are not included. Internal conversion and source-verification artifacts remain outside this publication directory.

## Structural information

The dataset is a release of processed **graph-to-label pairs**, not an anonymized or encrypted dataset. Molecular graph connectivity and atom/bond features retain structural information. Removing SMILES does not guarantee that structures cannot be identified or reconstructed. The owner has explicitly accepted publication of these processed pairs.

## Access and provenance

The complete processed data are supplied as separately prepared downloadable ZIP assets. Once uploaded by the owner to GitHub Releases, Figshare or Zenodo, the actual URL/DOI should be added to README and the manuscript availability statement. The preparation itself does not create an external hosting record or DOI. No fictional access URL is provided.

Training from the processed inputs does not require SMILES or the original CSVs. Recreating the original graph-generation process does require the original structural representation, which is not distributed. The code includes documented repairs and configuration differences; complete historical paper-score/checkpoint equivalence is not claimed by the current validation.

Actual executed data and code checks are in `VALIDATION.md`. File inventories and anonymous split counts are in `configs/reproduction_manifest.json` and the data package's `manifest.json`.

The current training entry reads train/validation and evaluates the test and external profiles when available. Existing data archives remain compatible and unchanged; no data re-export is required.
