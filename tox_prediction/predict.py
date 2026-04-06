import math

import torch
from torch_geometric.data import Data, Batch

from .features import smiles_to_graph
from .models import load_model


def predict_smiles(smiles_list, model_path, device="cpu"):
    """Load a saved model and predict LD50 for a list of SMILES strings.

    Returns a list of predicted values (float). Invalid SMILES get NaN.
    """
    device = torch.device(device)
    model, _ = load_model(model_path, device)
    model.eval()

    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        graph = smiles_to_graph(smi)
        if graph is not None:
            graphs.append(graph)
            valid_indices.append(i)

    results = [math.nan] * len(smiles_list)

    if graphs:
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            preds = model(batch.x, batch.edge_index, batch.batch).squeeze().cpu()

        if preds.dim() == 0:
            preds = preds.unsqueeze(0)

        for idx, pred in zip(valid_indices, preds):
            results[idx] = pred.item()

    return results
