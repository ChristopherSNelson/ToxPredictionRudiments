import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="Run TDC ADMET benchmark for LD50_Zhu")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds")
    parser.add_argument("--data-path", type=str, default="data/", help="Path for TDC data cache")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, or auto")
    args = parser.parse_args()

    from tox_prediction.evaluate_tdc import run_tdc_benchmark

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_tdc_benchmark(
        n_trials=args.n_trials,
        seeds=args.seeds,
        data_path=args.data_path,
        device=device,
    )


if __name__ == "__main__":
    main()
