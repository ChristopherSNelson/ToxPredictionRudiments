#import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tdc.single_pred import Tox
from tqdm import tqdm
import optuna

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load toxicity data
data = Tox(name='LD50_Zhu')
split = data.get_split()
train_df = split['train']
valid_df = split['valid']
test_df = split['test']

print(f"Train size: {len(train_df)}")
print(f"Valid size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")

# Sample some data to see the structure
print("\nSample data:")
print(train_df.head())

# Function to convert SMILES to graph data
def smiles_to_graph(smiles):
    """Convert SMILES string to graph data object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            # Features: atomic number, degree, formal charge, hybridization, aromaticity
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                int(atom.GetIsAromatic()),
                atom.GetChiralTag(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs()
            ]
            atom_features.append(features)
            
        # Convert to one-hot for categorical features
        x = torch.zeros(len(atom_features), 56)  # Increase feature size to accommodate one-hot encoding
        
        for i, features in enumerate(atom_features):
            # Atomic number (one-hot, assuming max atomic num = 36)
            if features[0] <= 36:
                x[i, features[0]-1] = 1
            
            # Degree (one-hot, 0-5)
            degree = min(features[1], 5)
            x[i, 36 + degree] = 1
            
            # Formal charge (shifted to one-hot, -1, 0, 1)
            fc = features[2] + 1  # Shift to 0,1,2
            if 0 <= fc <= 2:
                x[i, 42 + fc] = 1
                
            # Hybridization (one-hot)
            hyb = min(int(features[3]), 3)  # SP, SP2, SP3, Other
            x[i, 45 + hyb] = 1
            
            # Is Aromatic (binary)
            x[i, 49] = features[4]
            
            # Chirality (binary features)
            x[i, 50] = float(features[5] > 0)
            
            # Radical electrons (binary features)
            x[i, 51] = float(features[6] > 0)
            
            # Total hydrogens (one-hot, 0-4)
            h_count = min(features[7], 4)
            x[i, 52 + h_count] = 1
        
        # Get edge indices and features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_type = bond.GetBondTypeAsDouble()
            is_conj = int(bond.GetIsConjugated())
            is_in_ring = int(bond.IsInRing())
            
            # Add edges in both directions
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([[bond_type, is_conj, is_in_ring], [bond_type, is_conj, is_in_ring]])
            
        if len(edge_indices) == 0:  # Handle molecules with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data
    
    except Exception as e:
        print(f"Error in smiles_to_graph for {smiles}: {e}")
        return None

# Process the dataset
def process_dataset(dataframe):
    """Convert DataFrame with SMILES to list of graph Data objects."""
    data_list = []
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing molecules"):
        smiles = row['Drug']
        y = row['Y']
        
        graph_data = smiles_to_graph(smiles)
        if graph_data is not None:
            graph_data.y = torch.tensor([y], dtype=torch.float)
            graph_data.smiles = smiles  # Store SMILES for reference
            data_list.append(graph_data)
    
    return data_list

# Process datasets
print("\nProcessing train dataset...")
train_data = process_dataset(train_df)
print("\nProcessing validation dataset...")
valid_data = process_dataset(valid_df)
print("\nProcessing test dataset...")
test_data = process_dataset(test_df)

print(f"\nProcessed train size: {len(train_data)}")
print(f"Processed valid size: {len(valid_data)}")
print(f"Processed test size: {len(test_data)}")

# GNN Model Definitions
class GCNNet(nn.Module):
    """Graph Convolutional Network for toxicity prediction."""
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.2):
        super(GCNNet, self).__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_node_features, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output MLP
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        # Message passing layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP for final prediction
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x

class GATNet(nn.Module):
    """Graph Attention Network for toxicity prediction."""
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, heads=4, dropout=0.2):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        
        # Output MLP
        self.lin1 = nn.Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        # Message passing layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP for final prediction
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return x

# Training function
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(train_loader.dataset)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    mse = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            mse += F.mse_loss(out.squeeze(), data.y).item() * data.num_graphs
            predictions.extend(out.squeeze().cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    
    rmse = np.sqrt(mse / len(loader.dataset))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return rmse, mae, r2, predictions, targets

# Function for hyperparameter tuning with Optuna
def objective(trial):
    # Hyperparameters to tune
    model_type = trial.suggest_categorical('model_type', ['GAT']) #['GCN', 'GAT']
    hidden_channels = trial.suggest_int('hidden_channels', 32, 128, step=16)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.7, step=0.1)# ('dropout', 0.1, 0.5, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Additional hyperparameters for GAT
    heads = 1
    if model_type == 'GAT':
        heads = trial.suggest_int('heads', 2, 8)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    
    # Initialize model
    num_node_features = train_data[0].x.shape[1]
    
    if model_type == 'GCN':
        model = GCNNet(num_node_features=num_node_features, 
                      hidden_channels=hidden_channels,
                      num_layers=num_layers,
                      dropout=dropout).to(device)
    else:  # GAT
        model = GATNet(num_node_features=num_node_features, 
                      hidden_channels=hidden_channels,
                      num_layers=num_layers,
                      heads=heads,
                      dropout=dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Early stopping parameters
    patience = 10
    best_val_rmse = float('inf')
    counter = 0
    
    # Training loop
    for epoch in range(100):  # Max 100 epochs
        train_loss = train(model, train_loader, optimizer, device)
        val_rmse, val_mae, val_r2, _, _ = evaluate(model, valid_loader, device)
        
        # Trial pruning (stop unpromising trials early)
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    return best_val_rmse

# Hyperparameter tuning
def run_hyperparameter_tuning():
    print("\nStarting hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

# Train with best hyperparameters
def train_with_best_params(best_params):
    print("\nTraining model with best hyperparameters...")
    
    # Data loaders
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model
    num_node_features = train_data[0].x.shape[1]
    model_type = best_params['model_type']
    
    if model_type == 'GCN':
        model = GCNNet(num_node_features=num_node_features, 
                      hidden_channels=best_params['hidden_channels'],
                      num_layers=best_params['num_layers'],
                      dropout=best_params['dropout']).to(device)
    else:  # GAT
        model = GATNet(num_node_features=num_node_features, 
                      hidden_channels=best_params['hidden_channels'],
                      num_layers=best_params['num_layers'],
                      heads=best_params.get('heads', 4),
                      dropout=best_params['dropout']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    # Training with early stopping
    best_val_rmse = float('inf')
    best_model_state = None
    patience = 15
    counter = 0
    epochs = 200
    
    train_losses = []
    val_rmses = []
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_rmse, val_mae, val_r2, _, _ = evaluate(model, valid_loader, device)
        
        train_losses.append(train_loss)
        val_rmses.append(val_rmse)
        
        print(f"Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(model, test_loader, device)
    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_rmses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    
    # Create DataFrame with predictions vs actual values
    test_smiles = [data.smiles for data in test_data]
    results_df = pd.DataFrame({
        'SMILES': test_smiles,
        'Actual_LD50': test_targets,
        'Predicted_LD50': test_preds
    })
    
    # Save results to CSV
    results_df.to_csv('ld50_predictions.csv', index=False)
    print("\nSaved predictions to ld50_predictions.csv")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(test_targets, test_preds, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('Actual LD50')
    plt.ylabel('Predicted LD50')
    plt.title(f'Actual vs Predicted LD50 (R² = {test_r2:.4f})')
    plt.grid(True, alpha=0.3)
    plt.savefig('actual_vs_predicted.png')
    
    return model, results_df

def save_model(model, best_params, file_path='toxicity_model.pt'):
    """Save the trained model and its hyperparameters to disk."""
    # Create a dictionary containing everything needed to restore the model
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': best_params['model_type'],
        'hidden_channels': best_params['hidden_channels'],
        'num_layers': best_params['num_layers'],
        'dropout': best_params['dropout'],
        'num_node_features': model.conv_layers[0].in_channels,  # Save input feature dimension
    }

    # Add model-specific parameters
    if best_params['model_type'] == 'GAT':
        save_dict['heads'] = best_params.get('heads', 4)
    
    # Save to disk
    torch.save(save_dict, file_path)
    print(f"Model saved to {file_path}")


if __name__ == "__main__":
    # Run hyperparameter tuning
    best_params = run_hyperparameter_tuning()
    
    # Train with best hyperparameters
    model, results_df = train_with_best_params(best_params)

    save_model(model, best_params)
    
    print("\nExecution completed successfully!")
