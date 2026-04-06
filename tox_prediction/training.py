import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import optuna
from sklearn.metrics import mean_absolute_error, r2_score

from .models import build_model
from .data import make_loaders


def train_epoch(model, loader, optimizer, device):
    """Run one training epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate_model(model, loader, device):
    """Evaluate model on a DataLoader. Returns dict with rmse, mae, r2, predictions, targets."""
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

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
        "targets": targets,
    }


def _optuna_objective(trial, train_data, valid_data, device):
    """Optuna objective function. Minimizes validation RMSE."""
    model_type = trial.suggest_categorical("model_type", ["GCN", "GAT"])
    hidden_channels = trial.suggest_int("hidden_channels", 32, 128, step=16)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.7, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    heads = 1
    if model_type == "GAT":
        heads = trial.suggest_int("heads", 2, 8)

    train_loader, valid_loader, _ = make_loaders(train_data, valid_data, batch_size=batch_size)

    num_node_features = train_data[0].x.shape[1]
    model = build_model(model_type, num_node_features, hidden_channels, num_layers, dropout, heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    patience = 10
    best_val_rmse = float("inf")
    counter = 0

    for epoch in range(100):
        train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_model(model, valid_loader, device)
        val_rmse = metrics["rmse"]

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_val_rmse


def run_hpo(train_data, valid_data, device, n_trials=20):
    """Run Optuna hyperparameter optimization. Returns best params dict."""
    print("\nStarting hyperparameter tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _optuna_objective(trial, train_data, valid_data, device),
        n_trials=n_trials,
    )

    print("Best trial:")
    best = study.best_trial
    print(f"  Value (RMSE): {best.value}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    return best.params


def train_full(params, train_data, valid_data, test_data, device, max_epochs=200, patience=15):
    """Full training run with early stopping. Returns (model, test_metrics, train_losses, val_rmses)."""
    batch_size = params["batch_size"]
    train_loader, valid_loader, test_loader = make_loaders(train_data, valid_data, test_data, batch_size)

    num_node_features = train_data[0].x.shape[1]
    model = build_model(
        params["model_type"],
        num_node_features,
        params["hidden_channels"],
        params["num_layers"],
        params["dropout"],
        params.get("heads", 4),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    best_val_rmse = float("inf")
    best_model_state = None
    counter = 0
    train_losses = []
    val_rmses = []

    for epoch in range(max_epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_model(model, valid_loader, device)
        val_rmse = metrics["rmse"]

        train_losses.append(loss)
        val_rmses.append(val_rmse)

        print(
            f"Epoch: {epoch+1:03d}, Train Loss: {loss:.4f}, "
            f"Val RMSE: {val_rmse:.4f}, Val MAE: {metrics['mae']:.4f}, Val R2: {metrics['r2']:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"\nTest RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")

    return model, test_metrics, train_losses, val_rmses
