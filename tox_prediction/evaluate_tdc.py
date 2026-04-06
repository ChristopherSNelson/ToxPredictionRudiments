import numpy as np
import torch

from .data import process_dataset
from .training import run_hpo, train_full


def run_tdc_benchmark(n_trials=20, seeds=None, data_path="data/", device=None):
    """Run the TDC ADMET benchmark for LD50_Zhu across multiple seeds."""
    from tdc.benchmark_group import admet_group

    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    group = admet_group(path=data_path)
    predictions_list = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        benchmark = group.get("LD50_Zhu")
        name = benchmark["name"]
        train_val, test = benchmark["train_val"], benchmark["test"]
        train, valid = group.get_train_valid_split(benchmark=name, split_type="default", seed=seed)

        torch.manual_seed(seed)
        np.random.seed(seed)

        train_data = process_dataset(train)
        valid_data = process_dataset(valid)
        test_data = process_dataset(test)

        best_params = run_hpo(train_data, valid_data, device, n_trials=n_trials)
        model, test_metrics, _, _ = train_full(best_params, train_data, valid_data, test_data, device)

        y_pred_test = test_metrics["predictions"]

        predictions = {name: y_pred_test}
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    print(results)
    return results
