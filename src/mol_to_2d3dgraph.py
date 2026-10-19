import torch
import dgl
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from compound_tools import mol_to_geognn_graph_data_MMFF3d

__all__ = [
    "_add_virtual_nodes",
    "_apply_featurizers",
    "_create_atom_graph_from_mol",
    "_process_mol_base",
    "construct_3d_bigraph_from_mol",
    "construct_bigraph_from_mol",
    "safe_mol_to_geognn_graph_data_MMFF3d",
    "smiles_to_3dbigraph",
    "smiles_to_bigraph",
]


def _process_mol_base(mol, canonical_atom_order=True, explicit_hydrogens=False):
    if mol is None:
        return None
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    return mol


def _add_virtual_nodes(g, num_virtual_nodes):
    if num_virtual_nodes <= 0:
        return g
    num_real_nodes = g.num_nodes()
    real_nodes = list(range(num_real_nodes))
    g.add_nodes(num_virtual_nodes)
    virtual_src, virtual_dst = ([], [])
    for count in range(num_virtual_nodes):
        virtual_node = num_real_nodes + count
        virtual_node_copy = [virtual_node] * num_real_nodes
        virtual_src.extend(real_nodes)
        virtual_src.extend(virtual_node_copy)
        virtual_dst.extend(virtual_node_copy)
        virtual_dst.extend(real_nodes)
    if virtual_src:
        g.add_edges(
            torch.tensor(virtual_src, dtype=torch.int32),
            torch.tensor(virtual_dst, dtype=torch.int32),
        )
    for nk, nv in g.ndata.items():
        nv = torch.cat([nv, torch.zeros(num_virtual_nodes, nv.shape[1])], dim=0)
        nv[-num_virtual_nodes:, -1] = 1
        g.ndata[nk] = nv
    virtual_edges_count = num_virtual_nodes * num_real_nodes * 2
    for ek, ev in g.edata.items():
        ev = torch.cat([ev, torch.zeros(virtual_edges_count, ev.shape[1])], dim=0)
        ev[-virtual_edges_count:, -1] = 1
        g.edata[ek] = ev
    return g


def _apply_featurizers(g, mol, node_featurizer, edge_featurizer):
    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))
    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))
    return g


def construct_bigraph_from_mol(mol, add_self_loop=False):
    g = dgl.graph(([], []), idtype=torch.int32)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    src_list, dst_list = ([], [])
    for bond in mol.GetBonds():
        u, v = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        src_list.extend([u, v])
        dst_list.extend([v, u])
    if add_self_loop:
        nodes = list(range(num_atoms))
        src_list.extend(nodes)
        dst_list.extend(nodes)
    if src_list:
        g.add_edges(
            torch.tensor(src_list, dtype=torch.int32), torch.tensor(dst_list, dtype=torch.int32)
        )
    return g


def smiles_to_bigraph(
    smiles,
    add_self_loop=False,
    node_featurizer=None,
    edge_featurizer=None,
    canonical_atom_order=True,
    explicit_hydrogens=False,
    num_virtual_nodes=0,
):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = _process_mol_base(mol, canonical_atom_order, explicit_hydrogens)
        if mol is None:
            return None
        g = construct_bigraph_from_mol(mol, add_self_loop)
        g = _apply_featurizers(g, mol, node_featurizer, edge_featurizer)
        g = _add_virtual_nodes(g, num_virtual_nodes)
        return g
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None


def construct_3d_bigraph_from_mol(smiles, add_self_loop=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = safe_mol_to_geognn_graph_data_MMFF3d(mol)
        if data is None:
            return None
        g = dgl.graph(([], []), idtype=torch.int32)
        num_bonds = len(data["edges"])
        if num_bonds == 0:
            return _create_atom_graph_from_mol(mol, add_self_loop)
        g.add_nodes(num_bonds)
        bond_features = []
        for i, edge in enumerate(data["edges"]):
            bond_type = data["bond_type"][i] if i < len(data["bond_type"]) else 0
            is_in_ring = data["is_in_ring"][i] if i < len(data["is_in_ring"]) else 0
            bond_dir = data["bond_dir"][i] if i < len(data["bond_dir"]) else 0
            bond_length = data["bond_length"][i] if i < len(data["bond_length"]) else 0.0
            bond_features.append([bond_type, is_in_ring, bond_dir, bond_length])
        g.ndata["hv"] = torch.tensor(bond_features, dtype=torch.float32)
        if "BondAngleGraph_edges" in data and len(data["BondAngleGraph_edges"]) > 0:
            bond_angle_edges = data["BondAngleGraph_edges"]
            bond_angles = data["bond_angle"]
            if isinstance(bond_angle_edges, np.ndarray):
                bond_angle_edges = torch.from_numpy(bond_angle_edges).to(torch.int32)
            elif isinstance(bond_angle_edges, torch.Tensor):
                bond_angle_edges = bond_angle_edges.to(torch.int32)
            else:
                bond_angle_edges = torch.tensor(bond_angle_edges, dtype=torch.int32)
            if (
                bond_angle_edges.numel() > 0
                and bond_angle_edges.shape[0] > 0
                and (torch.max(bond_angle_edges) < num_bonds)
            ):
                g.add_edges(bond_angle_edges[:, 0], bond_angle_edges[:, 1])
                if bond_angles is not None and len(bond_angles) > 0:
                    if isinstance(bond_angles, np.ndarray):
                        if bond_angles.ndim == 1:
                            g.edata["he"] = torch.from_numpy(bond_angles).float().unsqueeze(1)
                        else:
                            g.edata["he"] = torch.from_numpy(bond_angles).float()
                    else:
                        try:
                            g.edata["he"] = torch.tensor(bond_angles, dtype=torch.float32)
                        except:
                            max_len = max(
                                (
                                    len(angle) if isinstance(angle, (list, np.ndarray)) else 1
                                    for angle in bond_angles
                                )
                            )
                            bond_angles_padded = []
                            for angle in bond_angles:
                                if isinstance(angle, (list, np.ndarray)):
                                    padded = list(angle) + [0] * (max_len - len(angle))
                                else:
                                    padded = [float(angle)] + [0] * (max_len - 1)
                                bond_angles_padded.append(padded)
                            g.edata["he"] = torch.tensor(bond_angles_padded, dtype=torch.float32)
        if "he" not in g.edata:
            g.edata["he"] = torch.zeros(g.num_edges(), 1, dtype=torch.float32)
        if add_self_loop:
            nodes = torch.arange(g.num_nodes(), dtype=torch.int32)
            g.add_edges(nodes, nodes)
        return g
    except Exception as e:
        print(f"Error processing 3D SMILES {smiles}: {e}")
        return None


def _create_atom_graph_from_mol(mol, add_self_loop=False):
    try:
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        g = dgl.graph(([], []), idtype=torch.int32)
        g.add_nodes(num_atoms)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(
                [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetIsAromatic()),
                ]
            )
        g.ndata["hv"] = torch.tensor(atom_features, dtype=torch.float32)
        g.edata["he"] = torch.zeros(g.num_edges(), 1, dtype=torch.float32)
        if add_self_loop:
            nodes = torch.arange(num_atoms, dtype=torch.int32)
            g.add_edges(nodes, nodes)
            g.edata["he"] = torch.zeros(num_atoms, 1, dtype=torch.float32)
        return g
    except Exception as e:
        print(f"Error creating atom graph: {e}")
        return None


def smiles_to_3dbigraph(smiles, add_self_loop=False, node_featurizer=None, edge_featurizer=None):
    try:
        g = construct_3d_bigraph_from_mol(smiles, add_self_loop)
        if g is None:
            return None
        return g
    except Exception as e:
        print(f"Error creating 3D graph for SMILES {smiles}: {e}")
        return None


def safe_mol_to_geognn_graph_data_MMFF3d(mol):
    try:
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        if data is None:
            return None
        if "edges" in data and data["edges"].dtype == np.int64:
            data["edges"] = data["edges"].astype(np.int32)
        if "BondAngleGraph_edges" in data and len(data["BondAngleGraph_edges"]) > 0:
            if data["BondAngleGraph_edges"].dtype == np.int64:
                data["BondAngleGraph_edges"] = data["BondAngleGraph_edges"].astype(np.int32)
        return data
    except Exception as e:
        print(f"Error in safe_mol_to_geognn_graph_data_MMFF3d: {e}")
        return None
