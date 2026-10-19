# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified: retained only the canonical feature dependency chain.
import itertools
from collections import defaultdict
from rdkit import Chem
import numpy as np
import torch
import dgl.backend as F

__all__ = [
    "BaseAtomFeaturizer",
    "BaseBondFeaturizer",
    "CanonicalAtomFeaturizer",
    "CanonicalBondFeaturizer",
    "ConcatFeaturizer",
    "atom_chirality_type_one_hot",
    "atom_degree_one_hot",
    "atom_formal_charge",
    "atom_hybridization_one_hot",
    "atom_implicit_valence_one_hot",
    "atom_is_aromatic",
    "atom_is_chiral_center",
    "atom_num_radical_electrons",
    "atom_total_num_H_one_hot",
    "atom_type_one_hot",
    "bond_is_conjugated",
    "bond_is_in_ring",
    "bond_stereo_one_hot",
    "bond_type_one_hot",
    "one_hot_encoding",
]


def one_hot_encoding(x, allowable_set, encode_unknown=False):
    if encode_unknown and allowable_set[-1] is not None:
        allowable_set.append(None)
    if encode_unknown and x not in allowable_set:
        x = None
    return list(map(lambda s: x == s, allowable_set))


def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "Si",
            "P",
            "Cl",
            "Br",
            "Mg",
            "Na",
            "Ca",
            "Fe",
            "As",
            "Al",
            "I",
            "B",
            "V",
            "K",
            "Tl",
            "Yb",
            "Sb",
            "Sn",
            "Ag",
            "Pd",
            "Co",
            "Se",
            "Ti",
            "Zn",
            "H",
            "Li",
            "Ge",
            "Cu",
            "Au",
            "Ni",
            "Cd",
            "In",
            "Mn",
            "Zr",
            "Cr",
            "Pt",
            "Hg",
            "Pb",
        ]
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)


def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)


def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)


def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)


def atom_formal_charge(atom):
    return [atom.GetFormalCharge()]


def atom_num_radical_electrons(atom):
    return [atom.GetNumRadicalElectrons()]


def atom_is_aromatic(atom):
    return [atom.GetIsAromatic()]


def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    if not atom.HasProp("_CIPCode"):
        return [False, False]
    if allowable_set is None:
        allowable_set = ["R", "S"]
    return one_hot_encoding(atom.GetProp("_CIPCode"), allowable_set, encode_unknown)


def atom_is_chiral_center(atom):
    return [atom.HasProp("_ChiralityPossible")]


class ConcatFeaturizer(object):
    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):
        return list(itertools.chain.from_iterable([func(x) for func in self.func_list]))


class BaseAtomFeaturizer(object):
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, (
                "feat_name should be provided if there are more than one features"
            )
            feat_name = list(self.featurizer_funcs.keys())[0]
        if feat_name not in self.featurizer_funcs:
            return ValueError(
                "Expect feat_name to be in {}, got {}".format(
                    list(self.featurizer_funcs.keys()), feat_name
                )
            )
        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))
        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))
        return processed_features


class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field="h"):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer(
                    [
                        atom_type_one_hot,
                        atom_degree_one_hot,
                        atom_implicit_valence_one_hot,
                        atom_formal_charge,
                        atom_is_aromatic,
                        atom_num_radical_electrons,
                        atom_hybridization_one_hot,
                        atom_is_aromatic,
                        atom_total_num_H_one_hot,
                        atom_is_chiral_center,
                        atom_chirality_type_one_hot,
                    ]
                )
            }
        )


def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)


def bond_is_conjugated(bond):
    return [bond.GetIsConjugated()]


def bond_is_in_ring(bond):
    return [bond.IsInRing()]


def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)


class BaseBondFeaturizer(object):
    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, (
                "feat_name should be provided if there are more than one features"
            )
            feat_name = list(self.featurizer_funcs.keys())[0]
        if feat_name not in self.featurizer_funcs:
            return ValueError(
                "Expect feat_name to be in {}, got {}".format(
                    list(self.featurizer_funcs.keys()), feat_name
                )
            )
        mol = Chem.MolFromSmiles("CCO")
        feats = self(mol)
        return feats[feat_name].shape[1]

    def __call__(self, mol):
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))
        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats
        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles("CO")
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats
        return processed_features


class CanonicalBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field="e", self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={
                bond_data_field: ConcatFeaturizer(
                    [bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot]
                )
            },
            self_loop=self_loop,
        )
