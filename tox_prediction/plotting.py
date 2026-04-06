import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, val_rmses, save_path=None):
    """Plot training loss and validation RMSE curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(val_rmses, label="Validation RMSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Validation RMSE")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved learning curves to {save_path}")
    return fig


def plot_actual_vs_predicted(targets, predictions, r2, save_path=None):
    """Plot actual vs predicted scatter with identity line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, predictions, alpha=0.5)
    ax.plot([min(targets), max(targets)], [min(targets), max(targets)], "r--")
    ax.set_xlabel("Actual LD50")
    ax.set_ylabel("Predicted LD50")
    ax.set_title(f"Actual vs Predicted LD50 (R2 = {r2:.4f})")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved plot to {save_path}")
    return fig
