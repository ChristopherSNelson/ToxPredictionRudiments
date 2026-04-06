import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Predict LD50 toxicity from SMILES using a trained model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .pt checkpoint")
    parser.add_argument("--smiles", type=str, nargs="+", help="One or more SMILES strings")
    parser.add_argument("--input-file", type=str, help="CSV file containing SMILES")
    parser.add_argument("--smiles-col", type=str, default="SMILES", help="Column name for SMILES in CSV")
    parser.add_argument("--output-file", type=str, help="Output CSV path (prints to stdout if omitted)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    from tox_prediction.predict import predict_smiles

    # Collect SMILES from args and/or file
    all_smiles = []

    if args.smiles:
        all_smiles.extend(args.smiles)

    if args.input_file:
        df = pd.read_csv(args.input_file)
        if args.smiles_col not in df.columns:
            print(f"Error: column '{args.smiles_col}' not found in {args.input_file}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        all_smiles.extend(df[args.smiles_col].tolist())

    if not all_smiles:
        print("Error: provide --smiles and/or --input-file", file=sys.stderr)
        sys.exit(1)

    # Predict
    predictions = predict_smiles(all_smiles, args.model_path, device=args.device)

    # Output
    results = pd.DataFrame({"SMILES": all_smiles, "Predicted_LD50": predictions})

    if args.output_file:
        results.to_csv(args.output_file, index=False)
        print(f"Saved {len(results)} predictions to {args.output_file}")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
