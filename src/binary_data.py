import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BinaryTrainingData:
    def __init__(self, directory):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["format_version"] != 1 or self.manifest["tasks"] != 135:
            raise ValueError("Unsupported binary dataset schema")
        self.profiles = {item["key"]: item for item in self.manifest["profiles"]}
        with np.load(self.root / self.manifest["graph_index"], allow_pickle=False) as saved:
            self.shard = saved["shard"].copy()
            self.position = saved["position"].copy()
        if self.shard.shape != self.position.shape or len(self.shard) != self.manifest["molecules"]:
            raise ValueError("Invalid binary graph index")
        with np.load(self.root / self.manifest["relations"], allow_pickle=False) as saved:
            self.relations = saved["relations"].astype(np.float32)
        if self.relations.shape != (135, 135) or not np.isin(self.relations, [0, 1]).all():
            raise ValueError("Invalid binary task relationship matrix")

    def load_profile(self, key, limit=None, seed=66):
        item = self.profiles[key]
        with np.load(self.root / item["path"], allow_pickle=False) as saved:
            if set(saved.files) != {"labels", "masks", "molecule_ids", "record_ids"}:
                raise ValueError("Unexpected split metadata fields")
            result = {name: saved[name].copy() for name in saved.files}
        rows = item["rows"]
        if result["labels"].shape != (rows, 135) or result["masks"].shape != (rows, 135):
            raise ValueError("Invalid binary label/mask dimensions")
        if result["molecule_ids"].shape != (rows,) or result["record_ids"].shape != (rows,):
            raise ValueError("Invalid anonymous ID dimensions")
        if not np.isfinite(result["labels"]).all() or not np.isin(result["masks"], [0, 1]).all():
            raise ValueError("Invalid labels or masks")
        observed = result["masks"][:, :108] != 0
        if not np.isin(result["labels"][:, :108][observed], [0, 1]).all():
            raise ValueError("Non-binary classification labels")
        ids = result["molecule_ids"]
        if np.any(ids < 0) or np.any(ids >= len(self.shard)):
            raise ValueError("Anonymous molecule ID outside graph index")
        if len(np.unique(result["record_ids"])) != rows:
            raise ValueError("Repeated anonymous record IDs")
        if limit is not None and limit < rows:
            selected = np.random.default_rng(seed).choice(rows, size=limit, replace=False)
            selected.sort()
            result = {name: value[selected] for name, value in result.items()}
        return result

    def verify_hashes(self):
        for item in tqdm(self.manifest["files"], desc="Dataset checksums"):
            path = self.root / item["path"]
            if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
                raise ValueError(f"Dataset file checksum mismatch: {item['path']}")


class BinaryGraphStore:
    def __init__(self, dataset, loading="preload", cache_shards=4):
        self.dataset = dataset
        self.loading = loading
        self.cache_shards = cache_shards
        self.loaded = {}
        self.cache = OrderedDict()

    def preload(self, molecule_ids):
        grouped = {}
        for molecule_id in np.unique(molecule_ids):
            gid = int(molecule_id)
            if gid not in self.loaded:
                shard = int(self.dataset.shard[gid])
                grouped.setdefault(shard, []).append(gid)
        for shard, ids in tqdm(
            grouped.items(), desc="Load cached graph inputs", disable=not sys.stderr.isatty()
        ):
            item = self.dataset.manifest["graph_shards"][shard]
            positions = [int(self.dataset.position[gid]) for gid in ids]
            count = item["molecules"]
            graphs, metadata = dgl.load_graphs(
                str(self.dataset.root / item["path"]),
                idx_list=positions + [count + pos for pos in positions],
            )
            if set(metadata) != {"molecule_ids"}:
                raise ValueError("Unexpected graph metadata; expected anonymous IDs only")
            if [int(metadata["molecule_ids"][pos]) for pos in positions] != ids:
                raise ValueError("Graph index and anonymous IDs disagree")
            for i, gid in enumerate(ids):
                self.loaded[gid] = (graphs[i], graphs[len(ids) + i])

    def get(self, molecule_id):
        gid = int(molecule_id)
        if gid in self.loaded:
            return self.loaded[gid]
        shard = int(self.dataset.shard[gid])
        position = int(self.dataset.position[gid])
        if shard not in self.cache:
            item = self.dataset.manifest["graph_shards"][shard]
            graphs, meta = dgl.load_graphs(str(self.dataset.root / item["path"]))
            if set(meta) != {"molecule_ids"}:
                raise ValueError("Unexpected graph metadata")
            self.cache[shard] = (graphs, meta, item["molecules"])
            while len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        graphs, metadata, count = self.cache[shard]
        if int(metadata["molecule_ids"][position]) != gid:
            raise ValueError("Graph index mismatch")
        return graphs[position], graphs[count + position]


class BinaryMoleculeDataset(Dataset):
    def __init__(self, profile, graph_store):
        self.profile = profile
        self.store = graph_store
        self.labels = torch.from_numpy(profile["labels"].astype(np.float32, copy=False))
        self.masks = torch.from_numpy(profile["masks"].astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        first, second = self.store.get(self.profile["molecule_ids"][index])
        record = int(self.profile["record_ids"][index])
        return f"R{record:09d}", first, second, self.labels[index], self.masks[index]
