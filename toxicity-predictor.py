import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
import joblib

class ToxicityPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def generate_features(self, smiles):
        """Generate molecular descriptors from SMILES strings"""
        features = []
        for smile in smiles:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    # If the SMILES string cannot be parsed, use placeholder values
                    features.append([0] * 1030)  # 1024 fingerprint bits + 6 descriptors
                    continue
                
                # Morgan fingerprints (ECFP)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp_array = np.zeros((1024,))
                DataStructs.ConvertToNumpyArray(morgan_fp, fp_array)
                
                # Basic molecular properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hba = Lipinski.NumHAcceptors(mol)
                hbd = Lipinski.NumHDonors(mol)
                rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                
                # Combine all features
                mol_features = list(fp_array) + [mw, logp, tpsa, hba, hbd, rotatable_bonds]
                features.append(mol_features)
            except:
                # Handle exceptions with placeholder values
                features.append([0] * 1030)  # 1024 fingerprint bits + 6 descriptors
        
        return np.array(features)
    
    def fit(self, smiles, toxicity_labels):
        """Train the model on SMILES strings and their toxicity labels"""
        X = self.generate_features(smiles)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, toxicity_labels)
        return self
    
    def predict(self, smiles):
        """Predict toxicity for new SMILES strings"""
        X = self.generate_features(smiles)
        return self.model.predict(X)
    
    def predict_proba(self, smiles):
        """Predict toxicity probability for new SMILES strings"""
        X = self.generate_features(smiles)
        return self.model.predict_proba(X)[:, 1]  # Probability of toxic class
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        self.model = joblib.load(filepath)
        return self
    
    def evaluate(self, smiles, true_labels):
        """Evaluate the model performance"""
        y_pred = self.predict(smiles)
        y_proba = self.predict_proba(smiles)
        
        accuracy = accuracy_score(true_labels, y_pred)
        auc = roc_auc_score(true_labels, y_proba)
        conf_matrix = confusion_matrix(true_labels, y_pred)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': conf_matrix
        }

# Example usage
def main():
    # Load your dataset
    # data = pd.read_csv('toxicity_dataset.csv')
    # smiles = data['SMILES'].tolist()
    # labels = data['Toxic'].tolist()
    
    # Example data for demonstration
    smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'C1=CC=C2C(=C1)C=CC=C2',      # Naphthalene
        'CC1=CC=C(C=C1)C(=C)C',       # Alpha-methylstyrene
        'CC1=C(C=CC=C1)C(=O)OCCN(C)C' # Procaine
    ]
    labels = [0, 1, 1, 0]  # Example labels: 0=non-toxic, 1=toxic
    
    # Split into training and test sets
    smiles_train, smiles_test, y_train, y_test = train_test_split(
        smiles, labels, test_size=0.2, random_state=42
    )
    
    # Train the model
    predictor = ToxicityPredictor()
    predictor.fit(smiles_train, y_train)
    
    # Evaluate
    results = predictor.evaluate(smiles_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    
    # Save the model
    predictor.save_model('toxicity_predictor.joblib')
    
    # Example prediction for a new compound
    new_smiles = 'CCC1=CC=CC=C1C(=O)OCCN(CC)CC'  # A fictional compound
    prediction = predictor.predict([new_smiles])[0]
    probability = predictor.predict_proba([new_smiles])[0]
    
    print(f"New compound prediction: {'Toxic' if prediction == 1 else 'Non-toxic'}")
    print(f"Probability of toxicity: {probability:.4f}")

if __name__ == "__main__":
    main()
