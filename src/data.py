import hashlib
import json
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_io import task_names
from featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from mol_to_2d3dgraph import smiles_to_bigraph, smiles_to_3dbigraph


class RegressionNormalizer:
    def __init__(self):
        self.means = np.zeros(27)
        self.stds = np.ones(27)
        self.fitted = False

    def fit(self, labels, masks):
        for i in range(27):
            values = labels[:, 108 + i][masks[:, 108 + i] != 0]
            if len(values):
                self.means[i] = np.mean(values)
                std = np.std(values)
                self.stds[i] = std if std > 0 else 1.0
        self.fitted = True
        return self

    def transform_labels(self, labels, masks, n_ctasks=108):
        if not self.fitted:
            raise ValueError("Fit the normalizer on training labels first")
        result = labels.clone() if torch.is_tensor(labels) else labels.copy()
        means = (
            torch.as_tensor(self.means, dtype=labels.dtype, device=labels.device)
            if torch.is_tensor(labels)
            else self.means
        )
        stds = (
            torch.as_tensor(self.stds, dtype=labels.dtype, device=labels.device)
            if torch.is_tensor(labels)
            else self.stds
        )
        regression = (result[:, n_ctasks:] - means) / stds
        mask = masks[:, n_ctasks:] != 0
        result[:, n_ctasks:] = (
            torch.where(mask, regression, result[:, n_ctasks:])
            if torch.is_tensor(labels)
            else np.where(mask, regression, result[:, n_ctasks:])
        )
        return result

    def inverse_transform_labels(self, normalized_labels, masks=None, n_ctasks=108):
        if not self.fitted:
            raise ValueError("Fit the normalizer before inverse transformation")
        result = (
            normalized_labels.clone()
            if torch.is_tensor(normalized_labels)
            else normalized_labels.copy()
        )
        start = n_ctasks if result.shape[1] == 135 else 0
        if result.shape[1] - start != 27:
            raise ValueError("Expected 27 regression columns or all 135 tasks")
        means = (
            torch.as_tensor(self.means, dtype=result.dtype, device=result.device)
            if torch.is_tensor(result)
            else self.means
        )
        stds = (
            torch.as_tensor(self.stds, dtype=result.dtype, device=result.device)
            if torch.is_tensor(result)
            else self.stds
        )
        values = result[:, start:] * stds + means
        if masks is not None:
            mask = masks[:, start:] != 0
            values = (
                torch.where(mask, values, result[:, start:])
                if torch.is_tensor(result)
                else np.where(mask, values, result[:, start:])
            )
        result[:, start:] = values
        return result

    def load_json(self, filename):
        values = json.loads(Path(filename).read_text(encoding="utf-8"))
        self.means = np.asarray(values["means"], dtype=np.float64)
        self.stds = np.asarray(values["stds"], dtype=np.float64)
        if (
            self.means.shape != (27,)
            or self.stds.shape != (27,)
            or not np.isfinite(self.means).all()
            or not np.isfinite(self.stds).all()
            or not (self.stds > 0).all()
            or not values["fitted"]
        ):
            raise ValueError("Invalid supplied fold normalization parameters")
        self.fitted = True
        return self

    def save(self, filename):
        Path(filename).write_text(
            json.dumps(
                {"means": self.means.tolist(), "stds": self.stds.tolist(), "fitted": self.fitted},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class MoleculeDataset(Dataset):
    def __init__(self, frame, cache_dir):
        self.smiles = frame.SMILES.tolist()
        values = frame[task_names()].to_numpy(dtype=np.float32)
        self.masks = torch.from_numpy((~np.isnan(values)).astype(np.float32))
        self.labels = torch.from_numpy(np.nan_to_num(values, nan=0.0))
        self.graphs = {}
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        atom = CanonicalAtomFeaturizer(atom_data_field="hv")
        bond = CanonicalBondFeaturizer(bond_data_field="he", self_loop=True)
        for smiles in tqdm(dict.fromkeys(self.smiles), desc="Molecular graphs"):
            key = hashlib.sha256(
                ("graph-v1-rdkit2023.9.6-conformer66:" + smiles).encode()
            ).hexdigest()
            cached = cache_dir / f"{key}.bin"
            if cached.exists():
                graphs, _ = dgl.load_graphs(str(cached))
                if len(graphs) != 2:
                    raise ValueError("Invalid graph cache; clear the selected cache directory")
                first, second = graphs
            else:
                first = smiles_to_bigraph(
                    smiles, add_self_loop=True, node_featurizer=atom, edge_featurizer=bond
                )
                second = smiles_to_3dbigraph(smiles)
                if first is None or second is None:
                    raise ValueError(
                        "Graph construction failed for an input SMILES; validate your local input with RDKit"
                    )
                if first.ndata["hv"].shape[1] != 78 or first.edata["he"].shape[1] != 13:
                    raise ValueError("Unexpected canonical graph feature dimensions")
                if second.ndata["hv"].shape[1] != 4 or second.edata["he"].shape[1] != 1:
                    raise ValueError("Unexpected 3D graph feature dimensions")
                dgl.save_graphs(str(cached), [first, second])
            self.graphs[smiles] = (first, second)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        first, second = self.graphs[smiles]
        return f"C{index:09d}", first, second, self.labels[index], self.masks[index]


def collate_molgraphs(data, normalizer):
    smiles, first, second, labels, masks = map(list, zip(*data))
    bg1, bg2 = dgl.batch(first), dgl.batch(second)
    for graph in [bg1, bg2]:
        graph.set_n_initializer(dgl.init.zero_initializer)
        graph.set_e_initializer(dgl.init.zero_initializer)
    labels, masks = torch.stack(labels), torch.stack(masks)
    return smiles, bg1, bg2, normalizer.transform_labels(labels, masks), masks
