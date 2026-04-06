import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from .features import NUM_NODE_FEATURES


class GCNNet(nn.Module):
    """Graph Convolutional Network for toxicity prediction."""

    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_node_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class GATNet(nn.Module):
    """Graph Attention Network for toxicity prediction."""

    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, heads=4, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        self.lin1 = nn.Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


def build_model(model_type, num_node_features, hidden_channels, num_layers, dropout, heads=4):
    """Factory function to create a GCN or GAT model."""
    if model_type == "GCN":
        return GCNNet(num_node_features, hidden_channels, num_layers, dropout)
    elif model_type == "GAT":
        return GATNet(num_node_features, hidden_channels, num_layers, heads, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def save_model(model, params, filepath="toxicity_model.pt"):
    """Save model weights and params needed to reconstruct it."""
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_type": params["model_type"],
        "hidden_channels": params["hidden_channels"],
        "num_layers": params["num_layers"],
        "dropout": params["dropout"],
        "num_node_features": model.conv_layers[0].in_channels,
    }
    if params["model_type"] == "GAT":
        save_dict["heads"] = params.get("heads", 4)

    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device=None):
    """Load a saved checkpoint and reconstruct the model.

    Returns (model, params_dict).
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model = build_model(
        model_type=checkpoint["model_type"],
        num_node_features=checkpoint["num_node_features"],
        hidden_channels=checkpoint["hidden_channels"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        heads=checkpoint.get("heads", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint
