import os
import pandas as pd


def find_best_epoch(log_path, metric_col="Val EMA (3-ep)"):
    """Read training log with restart detection, return (best_epoch, best_val, num_epochs)."""
    df = pd.read_csv(log_path)
    epochs = df["Epoch"].values

    blocks = []
    current_start = 0
    for i in range(1, len(epochs)):
        if epochs[i] <= epochs[i - 1]:
            blocks.append((current_start, i - 1))
            current_start = i
    blocks.append((current_start, len(epochs) - 1))

    main_block = max(blocks, key=lambda b: epochs[b[1]])
    block_df = df.iloc[main_block[0] : main_block[1] + 1]
    idx = block_df[metric_col].idxmax()
    best_epoch = int(block_df.loc[idx, "Epoch"])
    best_val = block_df.loc[idx, metric_col]
    return best_epoch, best_val, block_df


def best_epochs_from_logs(dataset="live", num_folds=5, metric="Val EMA (3-ep)"):
    results = []
    for fold in range(num_folds):
        log_path = f"{dataset}_fold{fold}_training_logs.csv"
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found")
            continue
        best_epoch, best_val, block_df = find_best_epoch(log_path, metric)
        best_row = block_df.loc[block_df[metric].idxmax()]
        results.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "val_pcc": best_row["Val PCC"],
            "val_srcc": best_row["Val SRCC"],
            "combined_val": best_row["Combined Val"],
            "val_ema": best_row["Val EMA (3-ep)"],
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="live")
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()

    df = best_epochs_from_logs(args.dataset, args.num_folds)
    print(df.to_string(index=False))
    print()
    print(f"Mean Combined: {df['combined_val'].mean():.4f}  ± {df['combined_val'].std():.4f}")
    print(f"Mean Val EMA:  {df['val_ema'].mean():.4f}  ± {df['val_ema'].std():.4f}")
