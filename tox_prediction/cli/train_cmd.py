import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description="Train a GNN toxicity model with Optuna HPO")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output files")
    parser.add_argument("--model-path", type=str, default="toxicity_model.pt", help="Model checkpoint filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")
    parser.add_argument("--cv", type=int, default=0, metavar="K", help="Run K-fold cross-validation (e.g. --cv 5)")
    args = parser.parse_args()

    from tox_prediction.data import load_tdc_ld50, process_dataset
    from tox_prediction.training import run_hpo, train_full, cross_validate
    from tox_prediction.models import save_model
    from tox_prediction.plotting import plot_learning_curves, plot_actual_vs_predicted

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data
    print("Loading TDC LD50_Zhu dataset...")
    train_df, valid_df, test_df = load_tdc_ld50()
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    train_data = process_dataset(train_df)
    valid_data = process_dataset(valid_df)
    test_data = process_dataset(test_df)

    # HPO
    best_params = run_hpo(train_data, valid_data, device, n_trials=args.n_trials)

    # Cross-validation (if requested)
    if args.cv > 1:
        all_data = train_data + valid_data + test_data
        cv_results = cross_validate(
            best_params, all_data, device,
            n_folds=args.cv, max_epochs=args.max_epochs, patience=args.patience,
        )

    # Train final model on full train/valid, evaluate on test
    model, test_metrics, train_losses, val_rmses = train_full(
        best_params, train_data, valid_data, test_data, device,
        max_epochs=args.max_epochs, patience=args.patience,
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    model_file = os.path.join(args.output_dir, args.model_path)
    save_model(model, best_params, model_file)

    plot_learning_curves(train_losses, val_rmses, os.path.join(args.output_dir, "learning_curves.png"))
    plot_actual_vs_predicted(
        test_metrics["targets"], test_metrics["predictions"],
        test_metrics["r2"], os.path.join(args.output_dir, "actual_vs_predicted.png"),
    )

    results_df = pd.DataFrame({
        "SMILES": [d.smiles for d in test_data],
        "Actual_LD50": test_metrics["targets"],
        "Predicted_LD50": test_metrics["predictions"],
    })
    csv_path = os.path.join(args.output_dir, "ld50_predictions.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    if args.cv > 1:
        print(f"\nCV ({args.cv}-fold): R2 = {cv_results['r2_mean']:.4f} +/- {cv_results['r2_std']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
