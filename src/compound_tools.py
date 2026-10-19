#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified: removed unused definitions and explanatory blocks.
# Original attribution: adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from compound_constants import DAY_LIGHT_FG_SMARTS_LIST


def rdchem_enum_to_list(values):
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


class CompoundKit(object):
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ["misc"],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "misc"],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "misc"],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "num_radical_e": [0, 1, 2, 3, 4, "misc"],
        "atom_is_in_ring": [0, 1],
        "valence_out_shell": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size3": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size4": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size5": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size6": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size7": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
        "in_num_ring_with_size8": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        "bond_stereo": rdchem_enum_to_list(rdchem.BondStereo.values),
        "is_conjugated": [0, 1],
    }
    atom_float_names = ["van_der_waals_radis", "partial_charge", "mass"]
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]
    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167
    period_table = Chem.GetPeriodicTable()

    @staticmethod
    def get_atom_value(atom, name):
        if name == "atomic_num":
            return atom.GetAtomicNum()
        elif name == "chiral_tag":
            return atom.GetChiralTag()
        elif name == "degree":
            return atom.GetDegree()
        elif name == "explicit_valence":
            return atom.GetExplicitValence()
        elif name == "formal_charge":
            return atom.GetFormalCharge()
        elif name == "hybridization":
            return atom.GetHybridization()
        elif name == "implicit_valence":
            return atom.GetImplicitValence()
        elif name == "is_aromatic":
            return int(atom.GetIsAromatic())
        elif name == "mass":
            return int(atom.GetMass())
        elif name == "total_numHs":
            return atom.GetTotalNumHs()
        elif name == "num_radical_e":
            return atom.GetNumRadicalElectrons()
        elif name == "atom_is_in_ring":
            return int(atom.IsInRing())
        elif name == "valence_out_shell":
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_bond_value(bond, name):
        if name == "bond_dir":
            return bond.GetBondDir()
        elif name == "bond_type":
            return bond.GetBondType()
        elif name == "is_in_ring":
            return int(bond.IsInRing())
        elif name == "is_conjugated":
            return int(bond.GetIsConjugated())
        elif name == "bond_stereo":
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts


class Compound3DKit(object):
    @staticmethod
    def get_atom_poses(mol, conf):
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs, randomSeed=66)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return (new_mol, atom_poses, energy)
        else:
            return (new_mol, atom_poses)

    @staticmethod
    def get_2d_atom_poses(mol):
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, "float32")
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type="HT"):
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-05)
            vec2 = vec2 / (norm2 + 1e-05)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == "HT":
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == "HH":
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])
        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], "int64")
            bond_angles = np.zeros([0], "float32")
        else:
            super_edges = np.array(super_edges, "int64")
            bond_angles = np.array(bond_angles, "float32")
        return (super_edges, bond_angles, bond_angle_dirs)


def mol_to_graph_data(mol):
    if len(mol.GetAtoms()) == 0:
        return None
    atom_id_names = [
        "atomic_num",
        "chiral_tag",
        "degree",
        "explicit_valence",
        "formal_charge",
        "hybridization",
        "implicit_valence",
        "is_aromatic",
        "total_numHs",
    ]
    bond_id_names = ["bond_dir", "bond_type", "is_in_ring"]
    data = {}
    for name in atom_id_names:
        data[name] = []
    data["mass"] = []
    for name in bond_id_names:
        data[name] = []
    data["edges"] = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)
        data["mass"].append(CompoundKit.get_atom_value(atom, "mass") * 0.01)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        data["edges"] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1
            data[name] += [bond_feature_id] * 2
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data["edges"] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2
        data[name] += [bond_feature_id] * N
    if len(data["edges"]) == 0:
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data["edges"] = np.zeros((0, 2), dtype="int64")
    for name in atom_id_names:
        data[name] = np.array(data[name], "int64")
    data["mass"] = np.array(data["mass"], "float32")
    for name in bond_id_names:
        data[name] = np.array(data[name], "int64")
    data["edges"] = np.array(data["edges"], "int64")
    data["morgan_fp"] = np.array(CompoundKit.get_morgan_fingerprint(mol), "int64")
    data["maccs_fp"] = np.array(CompoundKit.get_maccs_fingerprint(mol), "int64")
    data["daylight_fg_counts"] = np.array(
        CompoundKit.get_daylight_functional_group_counts(mol), "int64"
    )
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    if len(mol.GetAtoms()) == 0:
        return None
    data = mol_to_graph_data(mol)
    data["atom_pos"] = np.array(atom_poses, "float32")
    data["bond_length"] = Compound3DKit.get_bond_lengths(data["edges"], data["atom_pos"])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = Compound3DKit.get_superedge_angles(
        data["edges"], data["atom_pos"]
    )
    data["BondAngleGraph_edges"] = BondAngleGraph_edges
    data["bond_angle"] = np.array(bond_angles, "float32")
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type="HT")
