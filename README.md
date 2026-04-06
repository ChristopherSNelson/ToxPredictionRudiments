# Drug Toxicity Prediction (LD50)

Predicts acute oral toxicity (LD50) of drug compounds from SMILES strings using Graph Neural Networks. Uses the [TDC LD50_Zhu dataset](https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50) (~7,400 compounds).

## Models

- **GAT** (Graph Attention Network) - best results so far (R^2 ~0.52, RMSE ~0.66)
- **GCN** (Graph Convolutional Network) - simpler baseline
- **Random Forest** on Morgan fingerprints (`toxicity-predictor.py`) - early experiment

Hyperparameter tuning via [Optuna](https://optuna.org/).

## Project Structure

```
toxicity-predictor.py          # Random Forest baseline (Morgan fingerprints + descriptors)
drug_toxicity_prediction_v3.py # Main GNN script (GAT/GCN with Optuna tuning)
tdc_eval/                      # TDC benchmark evaluation
  drug_tox_eval.py
drug_pred_v1/                  # Earlier GNN iteration
goodGATrun/                    # Best GAT configuration (v2)
old_GNN/                       # Initial GNN attempt
```

## Setup

```bash
pip install -r requirements.txt
```

Data is automatically downloaded from TDC on first run.

## Usage

Train with hyperparameter tuning:

```bash
python drug_toxicity_prediction_v3.py
```

Run TDC benchmark evaluation:

```bash
cd tdc_eval
python drug_tox_eval.py
```

## Best Hyperparameters Found

| Parameter | Value |
|-----------|-------|
| Model | GAT |
| Hidden channels | 64 |
| Layers | 3 |
| Attention heads | 8 |
| Dropout | 0.2 |
| Learning rate | 0.002 |
| Batch size | 128 |
