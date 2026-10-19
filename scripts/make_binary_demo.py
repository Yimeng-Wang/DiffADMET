import argparse
import hashlib
import json
import sys
from pathlib import Path

import dgl
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import RegressionNormalizer
from featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from mol_to_2d3dgraph import smiles_to_bigraph, smiles_to_3dbigraph


def save_arrays(path, **arrays):
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Create an artificial anonymous-ID BIN dataset; no project research data read"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty demo output directory")
    for directory in ["graphs", "splits", "normalizers"]:
        (out / directory).mkdir(parents=True, exist_ok=True)
    molecules = [
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CO",
        "CCO",
        "CCCO",
        "CCCCO",
        "CN",
        "CCN",
        "CCCN",
        "CCOC",
        "CCOCC",
        "CC(=O)O",
        "CC(=O)N",
        "c1ccccc1",
        "Cc1ccccc1",
        "Oc1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCCC1",
        "CC(C)O",
        "CC(C)N",
    ]
    atom = CanonicalAtomFeaturizer(atom_data_field="hv")
    bond = CanonicalBondFeaturizer(bond_data_field="he", self_loop=True)
    first = [
        smiles_to_bigraph(value, add_self_loop=True, node_featurizer=atom, edge_featurizer=bond)
        for value in molecules
    ]
    second = [smiles_to_3dbigraph(value) for value in molecules]
    graph_path = out / "graphs/graphs_0000.bin"
    dgl.save_graphs(
        str(graph_path), first + second, {"molecule_ids": torch.arange(24, dtype=torch.int64)}
    )
    save_arrays(
        out / "graph_index.bin",
        shard=np.zeros(24, dtype=np.int32),
        position=np.arange(24, dtype=np.int32),
    )
    save_arrays(out / "relations.bin", relations=np.eye(135, dtype=np.uint8))
    rng = np.random.default_rng(66)
    values = rng.normal(size=(24, 135)).astype(np.float32)
    values[:, :108] = (np.arange(24)[:, None] + np.arange(108)[None, :]) % 2
    masks = np.ones_like(values, dtype=np.uint8)
    masks[0, 5] = 0
    masks[1, 112] = 0
    values[masks == 0] = 0
    normalizer = RegressionNormalizer().fit(values[:12], masks[:12])
    normalizers = []
    for fold in range(5):
        name = f"normalizers/fold_{fold}.json"
        normalizer.save(out / name)
        normalizers.append(name)
    profiles = []
    offset = 0
    entries = [
        (f"fold_{fold}/{split}", np.arange(12) if split == "train" else np.arange(12, 20))
        for fold in range(5)
        for split in ["train", "valid"]
    ]
    entries.extend([(name, np.arange(20, 24)) for name in ["test", "external"]])
    for key, ids in entries:
        rel = "splits/" + key.replace("/", "_") + ".bin"
        records = np.arange(offset, offset + len(ids), dtype=np.int64)
        offset += len(ids)
        save_arrays(
            out / rel,
            labels=values[ids],
            masks=masks[ids],
            molecule_ids=ids.astype(np.int64),
            record_ids=records,
        )
        profiles.append(
            {
                "key": key,
                "path": rel,
                "rows": len(ids),
                "source_rows": len(ids),
                "duplicates_removed": 0,
                "observed_labels": int(masks[ids].sum()),
                "bytes": (out / rel).stat().st_size,
                "sha256": digest(out / rel),
            }
        )
    files = [
        {
            "path": path.relative_to(out).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(out.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "format_version": 1,
        "project": "DiffADMET artificial demo",
        "tasks": 135,
        "molecules": 24,
        "graph_index": "graph_index.bin",
        "relations": "relations.bin",
        "graph_shards": [
            {
                "path": "graphs/graphs_0000.bin",
                "molecules": 24,
                "bytes": graph_path.stat().st_size,
                "sha256": digest(graph_path),
            }
        ],
        "profiles": profiles,
        "normalizers": normalizers,
        "files": files,
        "contains_plaintext_smiles": False,
        "note": "Artificial labels and identity relations. Demo split repeated across fold names; not a scientific cross-validation dataset.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Created artificial BIN inputs with numerical metadata only.")


if __name__ == "__main__":
    main()
