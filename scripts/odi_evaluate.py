#!/usr/bin/env python3
"""
Cross-dataset evaluation of LIVE-trained models on ODI.
Evaluates each fold's best checkpoint on ALL ODI images (zero-shot transfer).
"""

import os
import csv
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.utils import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_odi_data(dataset_name="odi"):
    """Load all ODI images and scores."""
    score_file = f"{dataset_name}_scores.csv"
    img_dir = dataset_name

    names, quality_scores = [], []
    with open(score_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["image_name"])
            quality_scores.append(float(row["quality_score"]))

    paths = []
    for name in names:
        for ext in ("", ".jpg", ".png", ".jpeg", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"):
            p = os.path.join(img_dir, name + ext)
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(os.path.join(img_dir, name))

    found = sum(1 for p in paths if os.path.exists(p))
    print(f"  Loaded {len(names)} image names, {found} files found on disk")
    return names, quality_scores, paths


def evaluate_model(model, dataset, batch_size=1, cpu_workers=4):
    """Run model on dataset, return z-score predictions."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=cpu_workers,
                        shuffle=False, collate_fn=collate_fn)
    preds_z = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference"):
            batch.pop("score")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits, _ = model(**batch, return_aux=True)
            preds_z.extend(logits.view(-1).cpu().tolist())
    return np.array(preds_z)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation on ODI")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--dataset", default="live", help="Source dataset (for checkpoint naming)")
    parser.add_argument("--target", default="odi", help="Target dataset for evaluation")
    args = parser.parse_args()

    if DEVICE == "cpu":
        print("WARNING: Running on CPU (will be slow)")

    names, quality_scores, paths = load_odi_data(args.target)
    gt_quality = np.array(quality_scores)

    hse_opts = {
        "patch_size": args.patch_size,
        "patch_stride": args.patch_size,
        "hse_grid_size": args.grid_size,
        "longer_side_lengths": [224, 384, 512],
        "max_seq_len_from_original_res": 0,
    }

    all_preds_raw = []
    fold_results = []

    for fold in range(args.num_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.num_folds}")
        print(f"{'='*60}")

        ckpt_path = f"{args.dataset}_fold{fold}_best_checkpoint.pth"
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        ema_sd = ckpt.get("model_state_dict") or ckpt.get("ema_state_dict")
        ckpt_mean = ckpt.get("global_mean", 0.0)
        ckpt_std = ckpt.get("global_std", 1.0)

        model = build_model(args.patch_size, args.grid_size, args.num_heads)
        model.load_state_dict(ema_sd)
        model.eval()

        # Normalize ODI quality scores using LIVE's stats (model's training stats)
        norm_scores = [(s - ckpt_mean) / ckpt_std for s in quality_scores]

        dataset = UnifiedODIQADataset(paths, norm_scores, hse_opts,
                                       augment=False, device="cpu",
                                       stereo=False)  # ODI is not stereo

        preds_z = evaluate_model(model, dataset, args.batch_size, args.workers)
        preds_raw = preds_z * ckpt_std + ckpt_mean

        # --- Per-fold metrics ---
        pcc = pearsonr(preds_raw, gt_quality)[0]
        srcc = spearmanr(preds_raw, gt_quality)[0]
        print(f"  PCC={pcc:.4f}  SRCC={srcc:.4f}")

        fold_results.append({
            "fold": fold,
            "pcc": float(pcc),
            "srcc": float(srcc),
            "preds_raw": preds_raw,
        })
        all_preds_raw.append(preds_raw)

    if not fold_results:
        print("No checkpoints found.")
        return

    # --- Per-fold summary ---
    print(f"\n{'='*60}")
    print("PER-FOLD ZERO-SHOT ON ODI")
    print(f"{'='*60}")
    print(f"{'Fold':<6} {'PCC':<10} {'SRCC':<10}")
    print("-" * 30)
    pccs, srccs = [], []
    for r in fold_results:
        print(f"{r['fold']:<6} {r['pcc']:<10.4f} {r['srcc']:<10.4f}")
        pccs.append(r["pcc"])
        srccs.append(r["srcc"])
    print("-" * 30)
    print(f"{'Mean':<6} {np.mean(pccs):<10.4f} {np.mean(srccs):<10.4f}")
    print(f"{'Std':<6} {np.std(pccs, ddof=1):<10.4f} {np.std(srccs, ddof=1):<10.4f}")

    # --- Ensemble (average predictions across folds) ---
    if len(all_preds_raw) > 1:
        ensemble_preds = np.mean(all_preds_raw, axis=0)
        ensemble_pcc = pearsonr(ensemble_preds, gt_quality)[0]
        ensemble_srcc = spearmanr(ensemble_preds, gt_quality)[0]
        print(f"\n{'='*60}")
        print(f"ENSEMBLE (avg of {len(fold_results)} folds)")
        print(f"{'='*60}")
        print(f"  PCC={ensemble_pcc:.4f}  SRCC={ensemble_srcc:.4f}")

        # Also report with 5-param logistic mapping
        from scipy.optimize import curve_fit
        def vqeg5pl(x, b1, b2, b3, b4, b5):
            return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5

        p0 = [np.ptp(gt_quality), 1.0, np.mean(ensemble_preds), 0.1, np.min(gt_quality)]
        try:
            popt, _ = curve_fit(vqeg5pl, ensemble_preds, gt_quality, p0=p0, maxfev=10000)
            mapped = vqeg5pl(ensemble_preds, *popt)
            nl_pcc = pearsonr(mapped, gt_quality)[0]
            print(f"  Nonlinear PLCC={nl_pcc:.4f}  (VQEG 5-param logistic)")
        except RuntimeError as e:
            print(f"  Nonlinear fit failed: {e}")

    # --- Save predictions ---
    for r in fold_results:
        fold = r["fold"]
        out_df = pd.DataFrame({
            "image_name": names,
            "fold": fold,
            "pred_raw": [f"{p:.6f}" for p in r["preds_raw"]],
            "gt_quality": [f"{s:.6f}" for s in gt_quality],
        })
        out_df.to_csv(f"odi_fold{fold}_predictions.csv", index=False)

    if len(fold_results) > 1:
        all_df = pd.concat([
            pd.DataFrame({
                "image_name": names,
                "fold": r["fold"],
                "pred_raw": [f"{p:.6f}" for p in r["preds_raw"]],
                "gt_quality": [f"{s:.6f}" for s in gt_quality],
            }) for r in fold_results
        ], ignore_index=True)
        all_df.to_csv("odi_kfold_predictions.csv", index=False)

    print(f"\nPredictions saved: odi_fold{{0..{args.num_folds-1}}}_predictions.csv + odi_kfold_predictions.csv")


if __name__ == "__main__":
    main()
