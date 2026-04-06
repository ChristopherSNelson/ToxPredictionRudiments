import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc.single_pred import Tox
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles_list = smiles_list
        self.labels = labels
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smile = self.smiles_list[idx]
        
        # Convert SMILES to molecular graph
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            # Handle invalid SMILES
            # Return a simple graph with one node as placeholder
            x = torch.zeros((1, 38))  # Changed to 38 features
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            
            if self.labels is not None:
                data.y = torch.tensor([self.labels[idx]], dtype=torch.float)
            return data
        
        # Get atom features
        atoms = mol.GetAtoms()
        x = []
        for atom in atoms:
            atom_features = []
            # One-hot encode atom type (C, N, O, F, etc.)
            atom_type = atom.GetAtomicNum()
            atom_type_onehot = [0] * 29  # Using 29 elements max
            atom_type_onehot[min(atom_type, 28)] = 1
            
            # Add other atom features
            atom_features.extend(atom_type_onehot)
            atom_features.append(atom.GetFormalCharge())
            atom_features.append(atom.GetNumExplicitHs())
            atom_features.append(atom.GetImplicitValence())
            atom_features.append(atom.GetIsAromatic())
            atom_features.append(atom.GetDegree())
            atom_features.append(atom.GetTotalDegree())
            atom_features.append(atom.GetTotalNumHs())
            # atom_features.append(atom.GetHybridization()) # Removing this as it might not be numeric
            
            x.append(atom_features)
            
        x = torch.tensor(x, dtype=torch.float)
        
        # Get bond connections (edges)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Add edges in both directions
            edge_index.append([i, j])
            edge_index.append([j, i])
            
        if len(edge_index) == 0:  # No bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Add label if available
        if self.labels is not None:
            data.y = torch.tensor([self.labels[idx]], dtype=torch.float)
            
        return data

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (single graph), create a batch with one graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Readout layer (graph-level pooling)
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        
        return x

class GraphToxicityPredictor:
    def __init__(self, input_dim=38, hidden_dim=64):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    def fit(self, train_loader, valid_loader=None, epochs=50, lr=0.001):
        """Train the GNN model"""
        self.model = GNN(self.input_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        train_losses = []
        valid_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                pred = self.model(batch)
                loss = loss_fn(pred, batch.y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch.num_graphs
                
            train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            if valid_loader is not None:
                valid_loss = self._evaluate(valid_loader, loss_fn)
                valid_losses.append(valid_loss)
                print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            else:
                print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}')
                
        return {"train_losses": train_losses, "valid_losses": valid_losses}
    
    def _evaluate(self, loader, loss_fn):
        """Evaluate the model on a data loader"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                loss = loss_fn(pred, batch.y)
                total_loss += loss.item() * batch.num_graphs
                
        return total_loss / len(loader.dataset)
    
    def predict(self, test_loader):
        """Make predictions on test data"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                predictions.extend(pred.cpu().numpy().flatten())
                
        return np.array(predictions)
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = GNN(self.input_dim, self.hidden_dim).to(self.device)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        return self
    
    def evaluate(self, test_loader, true_values):
        """Evaluate model performance"""
        predictions = self.predict(test_loader)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
    
    def plot_predictions(self, true_values, predictions, title="Predicted vs Actual Values"):
        """Plot predicted vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(true_values, predictions, alpha=0.5)
        
        # Add diagonal line
        min_val = min(min(true_values), min(predictions))
        max_val = max(max(true_values), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(title)
        plt.tight_layout()
        
        return plt

def collate_fn(data_list):
    return Batch.from_data_list(data_list)

def main():
    # Load the TDC toxicity dataset
    print("Loading LD50_Zhu dataset from TDC...")
    data = Tox(name='LD50_Zhu')
    split = data.get_split()
    
    train_df = split['train']
    valid_df = split['valid']
    test_df = split['test']
    
    print(f"Train set: {len(train_df)} compounds")
    print(f"Validation set: {len(valid_df)} compounds")
    print(f"Test set: {len(test_df)} compounds")
    
    # Debug: Check first few SMILES and their structures
    print("\nVerifying molecular structures from first 3 compounds:")
    for i, smile in enumerate(train_df['Drug'][:3]):
        print(f"SMILES {i+1}: {smile}")
        mol = Chem.MolFromSmiles(smile)
        if mol:
            print(f"  Valid molecule with {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
            # Debug a single molecule to confirm feature dimensions
            if i == 0:
                atoms = mol.GetAtoms()
                atom = atoms[0]
                atom_features = []
                atom_type = atom.GetAtomicNum()
                atom_type_onehot = [0] * 29
                atom_type_onehot[min(atom_type, 28)] = 1
                atom_features.extend(atom_type_onehot)
                atom_features.append(atom.GetFormalCharge())
                atom_features.append(atom.GetNumExplicitHs())
                atom_features.append(atom.GetImplicitValence())
                atom_features.append(atom.GetIsAromatic())
                atom_features.append(atom.GetDegree())
                atom_features.append(atom.GetTotalDegree())
                atom_features.append(atom.GetTotalNumHs())
                print(f"  Feature vector length: {len(atom_features)}")
        else:
            print("  Invalid SMILES string!")
    
    # Create PyG datasets with explicit input_dim
    train_dataset = MoleculeDataset(train_df['Drug'], train_df['Y'])
    valid_dataset = MoleculeDataset(valid_df['Drug'], valid_df['Y'])
    test_dataset = MoleculeDataset(test_df['Drug'], test_df['Y'])
    
    # Create data loaders with smaller batch size
    batch_size = 16  # Reduced from 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Verify input dimensions from an actual batch
    for batch in train_loader:
        print(f"\nInput feature dimensions: {batch.x.shape}")
        print(f"Expected input_dim: {batch.x.shape[1]}")
        break
    
    # Initialize and train the model with verified input_dim
    feature_dim = batch.x.shape[1]
    print(f"\nInitializing model with input dimension: {feature_dim}")
    predictor = GraphToxicityPredictor(input_dim=feature_dim)
    history = predictor.fit(train_loader, valid_loader, epochs=100)  # Reduced epochs for faster testing
    
    # Evaluate the model
    test_results = predictor.evaluate(test_loader, test_df['Y'].values)
    print(f"Test MSE: {test_results['mse']:.4f}")
    print(f"Test RMSE: {test_results['rmse']:.4f}")
    print(f"Test R²: {test_results['r2']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['valid_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    
    # Plot predictions vs actual values
    plot = predictor.plot_predictions(
        test_df['Y'].values, 
        test_results['predictions'],
        title='LD50 Predictions (Zhu Dataset)'
    )
    plot.savefig('predictions.png')
    
    # Save the model
    predictor.save_model('toxicity_gnn_model.pt')
    
    # Example prediction for a new compound
    new_smiles = 'CCC1=CC=CC=C1C(=O)OCCN(CC)CC'  # A fictional compound
    new_dataset = MoleculeDataset([new_smiles])
    new_loader = DataLoader(new_dataset, batch_size=1, collate_fn=collate_fn)
    prediction = predictor.predict(new_loader)[0]
    
    print(f"New compound LD50 prediction: {prediction:.4f}")

if __name__ == "__main__":
    main()
