import argparse
import json
import sys
from pathlib import Path

import dgl
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from binary_data import BinaryTrainingData
from data import RegressionNormalizer


def main():
    parser = argparse.ArgumentParser(
        description="Validate numerical BIN fields, anonymous IDs, graph correspondence, and published file hashes"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/reproduction"))
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    bundle = BinaryTrainingData(args.data_dir)
    if args.verify_hashes:
        bundle.verify_hashes()
    records = 0
    for key, item in bundle.profiles.items():
        profile = bundle.load_profile(key)
        labels, masks = profile["labels"], profile["masks"]
        assert int(masks.sum()) == item["observed_labels"]
        assert np.all(labels[masks == 0] == 0)
        assert np.all(bundle.shard[profile["molecule_ids"]] >= 0)
        assert np.unique(profile["molecule_ids"]).size == len(labels)
        records += len(labels)
        print(f"{key}: {len(labels)} records; observed labels={int(masks.sum())}")
    for normalizer in bundle.manifest["normalizers"]:
        RegressionNormalizer().load_json(bundle.root / normalizer)
    with np.load(bundle.root / bundle.manifest["graph_index"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"shard", "position"}
        assert all(arrays[name].dtype.kind in "biuf" for name in arrays.files)
    with np.load(bundle.root / bundle.manifest["relations"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"relations"}
    referenced = {item["path"] for item in bundle.manifest["files"]} | {"manifest.json"}
    actual = {
        path.relative_to(bundle.root).as_posix()
        for path in bundle.root.rglob("*")
        if path.is_file()
    }
    assert actual == referenced, "Unexpected or missing dataset files"
    graph_count = 0
    if args.deep:
        for shard, item in enumerate(bundle.manifest["graph_shards"]):
            graphs, metadata = dgl.load_graphs(str(bundle.root / item["path"]))
            assert set(metadata) == {"molecule_ids"}
            ids = metadata["molecule_ids"].numpy()
            count = item["molecules"]
            assert len(graphs) == count * 2 and len(ids) == count
            assert np.all(bundle.shard[ids] == shard)
            assert np.array_equal(bundle.position[ids], np.arange(count))
            for i, graph in enumerate(graphs):
                assert set(graph.ndata) == {"hv"} and set(graph.edata) == {"he"}
                dims = (78, 13) if i < count else (4, 1)
                assert graph.ndata["hv"].shape == (graph.num_nodes(), dims[0])
                assert graph.edata["he"].shape == (graph.num_edges(), dims[1])
                assert (
                    torch.isfinite(graph.ndata["hv"]).all()
                    and torch.isfinite(graph.edata["he"]).all()
                )
            graph_count += count
            if shard % 20 == 0:
                print(f"Validated graph shard {shard}: cumulative molecules={graph_count}")
        assert graph_count == bundle.manifest["molecules"]
    print(
        json.dumps(
            {
                "status": "passed",
                "molecules": bundle.manifest["molecules"],
                "records": records,
                "deep_graphs": graph_count,
                "plaintext_smiles_fields": 0,
                "model_weights": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
