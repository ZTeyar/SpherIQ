import os
import glob
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from spheriq.evaluate import load_model, read_scores, resolve_paths, collate_fn
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.splits import get_fold_split, get_scene_ids
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", "use_face_emb=True is set together")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "best_models_100"
NUM_FOLDS = 5
DATASET = "live"
TTA_ANGLES = [0]
BATCH_SIZE = 1
CPU_WORKERS = 4


def vqeg5pl(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5


def fit_plcc(pred, gt):
    p0 = [np.ptp(gt), 1.0, np.mean(pred), 0.1, np.min(gt)]
    try:
        popt, _ = curve_fit(vqeg5pl, pred, gt, p0=p0, maxfev=10000)
        mapped = vqeg5pl(pred, *popt)
        return pearsonr(mapped, gt)[0]
    except RuntimeError:
        return np.nan


def main():
    print(f"Device: {DEVICE}")

    image_names, quality_scores, _, _, _, _, _, _ = read_scores(f"{DATASET}_scores.csv")
    paths = resolve_paths(image_names, DATASET)
    all_scenes = get_scene_ids(DATASET, f"{DATASET}_scores.csv")

    hse_opts = {
        "patch_size": 32,
        "patch_stride": 32,
        "hse_grid_size": 10,
        "longer_side_lengths": [224, 384, 512],
        "max_seq_len_from_original_res": 0,
    }

    oof_preds = np.full(len(image_names), np.nan)
    fold_metrics = []

    print("\n" + "=" * 60)
    print("MEAN MODE — each fold evaluated on its held-out val scenes")
    print("=" * 60)

    for fold in range(NUM_FOLDS):
        _, val_scenes = get_fold_split(DATASET, f"{DATASET}_scores.csv", NUM_FOLDS, fold)
        val_idx = [i for i, s in enumerate(all_scenes) if s in val_scenes]
        if not val_idx:
            print(f"  Fold {fold}: no validation images found")
            continue

        ckpt = f"{MODEL_DIR}/{DATASET}_fold{fold}_epoch*_best.pth"
        files = glob.glob(ckpt)
        if not files:
            print(f"  Fold {fold}: checkpoint not found")
            continue
        path = files[0]

        val_paths = [paths[i] for i in val_idx]
        val_scores = [quality_scores[i] for i in val_idx]
        val_ds = UnifiedODIQADataset(val_paths, val_scores, hse_opts, augment=False, device="cpu", stereo=True)

        model, g_mean, g_std = load_model(path, patch_size=32, device=DEVICE, grid_size=10)
        fold_name = path.split("/")[-1]
        print(f"\nFold {fold} [{fold_name}] — {len(val_idx)} val images")

        all_tta = []
        for yaw in TTA_ANGLES:
            val_ds.yaw = yaw
            loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=CPU_WORKERS, shuffle=False, collate_fn=collate_fn)
            batch_preds = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"  Yaw {yaw}", leave=False):
                    batch.pop("score")
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    logits, _ = model(**batch, return_aux=True)
                    batch_preds.extend(logits.view(-1).cpu().tolist())
            del loader
            if g_mean is not None and g_std is not None:
                batch_preds = [p * g_std + g_mean for p in batch_preds]
            all_tta.append(batch_preds)

        preds = np.mean(all_tta, axis=0)
        pcc = pearsonr(preds, val_scores)[0]
        srcc = spearmanr(preds, val_scores)[0]
        nl_pcc = fit_plcc(preds, val_scores)
        fold_metrics.append((pcc, srcc, nl_pcc))
        nl_str = f"  NL_PCC={nl_pcc:.4f}" if not np.isnan(nl_pcc) else "  NL_PCC=fit_failed"
        print(f"  PCC={pcc:.4f}  {nl_str}  SRCC={srcc:.4f}")

        oof_preds[val_idx] = preds

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("MEAN MODE SUMMARY")
    print("=" * 60)
    pccs = [m[0] for m in fold_metrics]
    srccs = [m[1] for m in fold_metrics]
    nl_pccs = [m[2] for m in fold_metrics]
    for i, (p, s, nl) in enumerate(fold_metrics):
        nl_str = f"{nl:.4f}" if not np.isnan(nl) else "fit_failed"
        print(f"  Fold {i}:  PCC={p:.4f}  NL_PCC={nl_str}  SRCC={s:.4f}")
    print(f"  Mean ± Std:  PCC={np.mean(pccs):.4f} ± {np.std(pccs):.4f}  "
          f"NL_PCC={np.nanmean(nl_pccs):.4f} ± {np.nanstd(nl_pccs):.4f}  "
          f"SRCC={np.mean(srccs):.4f} ± {np.std(srccs):.4f}")

    valid_mask = ~np.isnan(oof_preds)
    oof_preds_arr = np.array(oof_preds)[valid_mask]
    oof_gt_arr = np.array(quality_scores)[valid_mask]
    oof_pcc = pearsonr(oof_preds_arr, oof_gt_arr)[0]
    oof_srcc = spearmanr(oof_preds_arr, oof_gt_arr)[0]
    oof_nl_pcc = fit_plcc(oof_preds_arr, oof_gt_arr)

    print("\n" + "=" * 60)
    print("ENSEMBLE MODE — out-of-fold predictions (OOF ensemble)")
    print("=" * 60)
    oof_nl_str = f"  NL_PCC={oof_nl_pcc:.4f}" if not np.isnan(oof_nl_pcc) else "  NL_PCC=fit_failed"
    print(f"  OOF PCC={oof_pcc:.4f}  {oof_nl_str}  SRCC={oof_srcc:.4f}  ({int(valid_mask.sum())}/{len(image_names)} images)")

    out = pd.DataFrame({"image_name": image_names, "oof_pred": oof_preds, "gt": quality_scores})
    out.to_csv(f"{DATASET}_best100_oof.csv", index=False)
    print(f"\nSaved {DATASET}_best100_oof.csv")


if __name__ == "__main__":
    main()
