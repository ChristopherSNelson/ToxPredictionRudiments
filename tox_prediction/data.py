import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from .features import smiles_to_graph


def load_tdc_ld50():
    """Load the TDC LD50_Zhu dataset and return (train_df, valid_df, test_df)."""
    from tdc.single_pred import Tox

    data = Tox(name="LD50_Zhu")
    split = data.get_split()
    return split["train"], split["valid"], split["test"]


def process_dataset(dataframe, smiles_col="Drug", y_col="Y"):
    """Convert a DataFrame of SMILES + targets to a list of PyG Data objects."""
    data_list = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing molecules"):
        smiles = row[smiles_col]
        y = row[y_col]

        graph_data = smiles_to_graph(smiles)
        if graph_data is not None:
            graph_data.y = torch.tensor([y], dtype=torch.float)
            graph_data.smiles = smiles
            data_list.append(graph_data)

    return data_list


def make_loaders(train_data, valid_data, test_data=None, batch_size=64):
    """Create PyG DataLoaders from processed datasets."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size) if test_data else None
    return train_loader, valid_loader, test_loader
