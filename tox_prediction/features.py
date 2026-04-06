import logging

import torch
from rdkit import Chem
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

NUM_NODE_FEATURES = 56
NUM_EDGE_FEATURES = 3


def smiles_to_graph(smiles):
    """Convert a SMILES string to a PyG Data object with 56-dim node features and 3-dim edge features.

    Returns None if the SMILES cannot be parsed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Extract raw atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                int(atom.GetIsAromatic()),
                atom.GetChiralTag(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(),
            ]
            atom_features.append(features)

        # One-hot encode into 56-dim vector
        x = torch.zeros(len(atom_features), NUM_NODE_FEATURES)

        for i, feat in enumerate(atom_features):
            # Atomic number (one-hot, 1-36)
            if feat[0] <= 36:
                x[i, feat[0] - 1] = 1

            # Degree (one-hot, 0-5)
            degree = min(feat[1], 5)
            x[i, 36 + degree] = 1

            # Formal charge (shifted one-hot: -1,0,1 -> indices 42,43,44)
            fc = feat[2] + 1
            if 0 <= fc <= 2:
                x[i, 42 + fc] = 1

            # Hybridization (one-hot, 0-3)
            hyb = min(int(feat[3]), 3)
            x[i, 45 + hyb] = 1

            # Is aromatic (binary)
            x[i, 49] = feat[4]

            # Chirality (binary)
            x[i, 50] = float(feat[5] > 0)

            # Radical electrons (binary)
            x[i, 51] = float(feat[6] > 0)

            # Total hydrogens (one-hot, 0-4)
            h_count = min(feat[7], 4)
            x[i, 52 + h_count] = 1

        # Edge indices and features
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feat = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]

            # Undirected: add both directions
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_feat, bond_feat])

        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, NUM_EDGE_FEATURES), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        logger.warning("Failed to convert SMILES '%s': %s", smiles, e)
        return None
