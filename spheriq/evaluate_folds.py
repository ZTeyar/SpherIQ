#!/usr/bin/env python3
"""
Per-fold evaluation with best-epoch selection by Combined Val.
For each fold: finds best epoch, restores best_checkpoint.pth, evaluates,
reports PLCC and SRCC, and computes final k-fold summary.
"""

import os
import csv
import math
import warnings
import copy
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from spheriq.musiq_arch import MUSIQ
from spheriq.splits import get_fold_split, get_scene_ids
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.utils import collate_fn, display_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_stable_seed(input_string):
    hash_object = hashlib.md5(input_string.encode("utf-8"))
    return int(hash_object.hexdigest(), 16) % (2 ** 32)


def get_fold_val_scenes(dataset_name="live", num_folds=5, fold_index=0):
    """Recompute the validation scene list for a given fold (delegates to splits)."""
    _, val_scenes = get_fold_split(dataset_name, f"{dataset_name}_scores.csv", num_folds, fold_index)
    return sorted(val_scenes)


def find_best_epoch(log_path, metric_col="Combined Val"):
    """Read training log, detect restart, return (best_epoch, best_val, block_df)."""
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


def load_per_epoch_checkpoint(ckpt_path, device=DEVICE):
    """Load per-epoch checkpoint, return (ema_state_dict, global_mean, global_std)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema_sd = ckpt.get("ema_state_dict")
    g_mean = ckpt.get("global_mean", 0.0)
    g_std = ckpt.get("global_std", 1.0)
    return ema_sd, g_mean, g_std


def build_model(patch_size=32, grid_size=10, num_heads=4, device=DEVICE):
    model = MUSIQ(
        patch_size=patch_size,
        num_class=1,
        use_spherical_coords=True,
        use_face_emb=True,
        use_scale_emb=True,
        pretrained=False,
        num_faces=6,
        spatial_pos_grid_size=grid_size,
        num_heads=num_heads,
        dropout_rate=0.1,
        attention_dropout_rate=0.0,
        longer_side_lengths=[224, 384, 512],
        max_seq_len_from_original_res=0,
    )
    return model.to(device)


def evaluate_fold(fold_index, dataset_name="live", num_folds=5,
                  tta_yaw_angles=None, batch_size=1, cpu_workers=4,
                  patch_size=32, grid_size=10, num_heads=4,
                  save_best_ckpt=True):
    """
    Evaluate a single fold:
      - Find best epoch by Combined Val from training logs
      - Load per-epoch checkpoint for that epoch
      - Restore best_checkpoint.pth
      - Run evaluation on held-out val scenes
      - Return (pcc, srcc, best_epoch, best_combined)
    """
    if tta_yaw_angles is None:
        tta_yaw_angles = [0]

    name_str = f"{dataset_name}_fold{fold_index}"
    log_path = f"{name_str}_training_logs.csv"
    ckpt_dir = f"{name_str}_checkpoints"
    best_ckpt_path = f"{name_str}_best_checkpoint.pth"

    if not os.path.exists(log_path):
        print(f"[Fold {fold_index}] Log not found: {log_path}")
        return None

    best_epoch, best_val, block_df = find_best_epoch(log_path)
    print(f"[Fold {fold_index}] Best epoch: {best_epoch} (Combined Val={best_val:.4f})")

    ckpt_file = os.path.join(ckpt_dir, f"checkpoint_{best_epoch}.pth")
    if not os.path.exists(ckpt_file):
        print(f"[Fold {fold_index}] Checkpoint not found: {ckpt_file}")
        return None

    ema_sd, ckpt_mean, ckpt_std = load_per_epoch_checkpoint(ckpt_file)
    if ema_sd is None:
        print(f"[Fold {fold_index}] No ema_state_dict in checkpoint, falling back to model_state_dict")
        ckpt = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
        ema_sd = ckpt.get("model_state_dict")
        if ema_sd is None:
            print(f"[Fold {fold_index}] No state_dict found at all")
            return None

    model = build_model(patch_size=patch_size, grid_size=grid_size, num_heads=num_heads)
    model.load_state_dict(ema_sd)
    model.eval()

    if save_best_ckpt:
        torch.save({
            "model_state_dict": ema_sd,
            "global_mean": ckpt_mean,
            "global_std": ckpt_std,
        }, best_ckpt_path)
        print(f"[Fold {fold_index}] Saved {best_ckpt_path}")

    val_scenes = get_fold_val_scenes(dataset_name, num_folds, fold_index)
    print(f"[Fold {fold_index}] Val scenes: {val_scenes}")

    hse_opts = {
        "patch_size": patch_size,
        "patch_stride": patch_size,
        "hse_grid_size": grid_size,
        "longer_side_lengths": [224, 384, 512],
        "max_seq_len_from_original_res": 0,
    }

    image_names, quality_scores, raw_scores, ds_means, ds_stds, ds_mins, ds_maxs, score_types = read_scores_with_paths(f"{dataset_name}_scores.csv")
    paths = resolve_paths(image_names, dataset_name)
    scenes = get_scene_ids(dataset_name, f"{dataset_name}_scores.csv")

    val_indices = []
    val_image_names = []
    for i, name in enumerate(image_names):
        if scenes[i] in val_scenes:
            val_indices.append(i)
            val_image_names.append(name)

    val_paths = [paths[i] for i in val_indices]
    val_scores = [quality_scores[i] for i in val_indices]

    if ckpt_mean is not None and ckpt_std is not None and (abs(ckpt_mean) > 1e-6 or abs(ckpt_std - 1.0) > 1e-6):
        norm_scores = [(s - ckpt_mean) / ckpt_std for s in val_scores]
        print(f"  Normalizing with μ={ckpt_mean:.4f}, σ={ckpt_std:.4f}")
    else:
        norm_scores = val_scores

    dataset = UnifiedODIQADataset(val_paths, norm_scores, hse_opts, augment=False, device="cpu", stereo=True)
    all_preds_z = []

    for yaw in tta_yaw_angles:
        dataset.yaw = yaw
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=cpu_workers, shuffle=False, collate_fn=collate_fn)

        batch_preds_z = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Fold {fold_index} TTA yaw={yaw}"):
                batch.pop("score")
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits, _ = model(**batch, return_aux=True)
                batch_preds_z.extend(logits.view(-1).cpu().tolist())

        del dataloader
        all_preds_z.append(batch_preds_z)

    preds_z = np.mean(all_preds_z, axis=0)
    preds_raw = [p * ckpt_std + ckpt_mean for p in preds_z] if (ckpt_mean is not None and ckpt_std is not None) else preds_z

    val_pcc = pearsonr(preds_z, norm_scores)[0]
    val_srcc = spearmanr(preds_z, norm_scores)[0]

    print(f"[Fold {fold_index}]  PCC={val_pcc:.4f}  SRCC={val_srcc:.4f}")

    best_row = block_df.loc[block_df["Epoch"] == best_epoch].iloc[0]
    logged_pcc = best_row["Val PCC"]
    logged_srcc = best_row["Val SRCC"]
    logged_combined = best_row["Combined Val"]
    print(f"             Logged: PCC={logged_pcc:.4f} SRCC={logged_srcc:.4f} Combined={logged_combined:.4f}")

    out_df = pd.DataFrame({
        "image_name": val_image_names,
        "fold": fold_index,
        "pred_z": [f"{p:.6f}" for p in preds_z],
        "pred_raw": [f"{p:.6f}" for p in preds_raw],
        "gt_quality": [f"{s:.6f}" for s in val_scores],
        "gt_norm": [f"{s:.6f}" for s in norm_scores],
    })
    pred_csv = f"{name_str}_predictions.csv"
    out_df.to_csv(pred_csv, index=False)
    print(f"             Saved {pred_csv}")

    return {
        "fold": fold_index,
        "best_epoch": best_epoch,
        "pcc": float(val_pcc),
        "srcc": float(val_srcc),
        "logged_pcc": float(logged_pcc),
        "logged_srcc": float(logged_srcc),
        "logged_combined": float(logged_combined),
        "predictions": out_df,
    }


def read_scores_with_paths(csv_path):
    image_names, quality_scores, raw_scores = [], [], []
    ds_means, ds_stds, ds_mins, ds_maxs, score_types = [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_names.append(row["image_name"])
            quality_scores.append(float(row["quality_score"]))
            raw_scores.append(float(row.get("raw_score", row["quality_score"])))
            ds_means.append(float(row.get("dataset_mean", 0.0)))
            ds_stds.append(float(row.get("dataset_std", 1.0)))
            has_min = "dataset_min" in row
            ds_mins.append(float(row["dataset_min"]) if has_min else None)
            ds_maxs.append(float(row["dataset_max"]) if has_min else None)
            score_types.append(row.get("score_type", "DMOS"))
    return image_names, quality_scores, raw_scores, ds_means, ds_stds, ds_mins, ds_maxs, score_types


def resolve_paths(image_names, folder):
    paths = []
    for name in image_names:
        for ext in ("", ".jpg", ".png", ".jpeg", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"):
            p = os.path.join(folder, name + ext)
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(os.path.join(folder, name))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate all k-fold models and restore best checkpoints")
    parser.add_argument("--dataset", default="live", help="Dataset name")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--tta", type=int, nargs="+", default=[0], help="TTA yaw angles")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--no-save", action="store_true", help="Skip saving best_checkpoint.pth files")
    args = parser.parse_args()

    if DEVICE == "cpu":
        print("WARNING: Running on CPU (will be slow)")

    results = []
    for fold in range(args.num_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.num_folds}")
        print(f"{'='*60}")
        r = evaluate_fold(
            fold_index=fold,
            dataset_name=args.dataset,
            num_folds=args.num_folds,
            tta_yaw_angles=args.tta,
            batch_size=args.batch_size,
            cpu_workers=args.workers,
            patch_size=args.patch_size,
            grid_size=args.grid_size,
            num_heads=args.num_heads,
            save_best_ckpt=not args.no_save,
        )
        if r is not None:
            results.append(r)

    if not results:
        print("\nNo results to report.")
        return

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Fold':<6} {'Epoch':<7} {'PCC':<9} {'SRCC':<9} {'Logged PCC':<12} {'Logged SRCC':<12} {'Logged Comb':<12}")
    print("-" * 60)
    all_pcc, all_srcc = [], []
    for r in results:
        print(f"{r['fold']:<6} {r['best_epoch']:<7} {r['pcc']:<9.4f} {r['srcc']:<9.4f} {r['logged_pcc']:<12.4f} {r['logged_srcc']:<12.4f} {r['logged_combined']:<12.4f}")
        all_pcc.append(r["pcc"])
        all_srcc.append(r["srcc"])

    print("-" * 60)
    pcc_arr, srcc_arr = np.array(all_pcc), np.array(all_srcc)
    print(f"{'Mean':<6} {'':<7} {pcc_arr.mean():<9.4f} {srcc_arr.mean():<9.4f}")
    print(f"{'Std':<6} {'':<7} {pcc_arr.std():<9.4f} {srcc_arr.std():<9.4f}")

    all_preds = pd.concat([r["predictions"] for r in results], ignore_index=True)
    all_preds.to_csv(f"{args.dataset}_kfold_predictions.csv", index=False)
    print(f"\nSaved {args.dataset}_kfold_predictions.csv ({len(all_preds)} rows)")

    ensemble_plcc = pearsonr(pcc_arr, srcc_arr)[0]
    print(f"\nPCC vs SRCC correlation across folds: {ensemble_plcc:.4f}")


if __name__ == "__main__":
    main()
