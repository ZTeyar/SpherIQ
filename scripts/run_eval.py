#!/usr/bin/env python3
"""
Full re-evaluation: OOF per-fold predictions + cross-dataset matrix.
Reads models from ../best_models_100/, saves to ../output/.
"""
import os, csv, glob, warnings
warnings.filterwarnings("ignore", message="use_face_emb=True is set together")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.utils import collate_fn
from spheriq.splits import get_fold_split, get_scene_ids

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = os.path.dirname(os.path.dirname(__file__))
BASE = REPO
MODEL_DIR = os.path.join(REPO, 'best_models_100')
OUT = os.path.join(REPO, 'output')
os.makedirs(OUT, exist_ok=True)

NUM_FOLDS = 5
GRID_SIZE = 10
PATCH_SIZE = 32
NUM_HEADS = 4
BATCH_SIZE = 1
CPU_WORKERS = 8
TTA_ANGLES = [0]

DATASET_CFG = {
    'live': {'stereo': True,  'csv': 'live_scores.csv',  'dir': 'live'},
    'cviq': {'stereo': False, 'csv': 'cviq_scores.csv',  'dir': 'cviq'},
    'jufe': {'stereo': False, 'csv': 'jufe_scores.csv',  'dir': 'jufe'},
    'odi':  {'stereo': False, 'csv': 'odi_scores.csv',   'dir': 'odi'},
}

hse_opts = {
    'patch_size': PATCH_SIZE, 'patch_stride': PATCH_SIZE,
    'hse_grid_size': GRID_SIZE,
    'longer_side_lengths': [224, 384, 512],
    'max_seq_len_from_original_res': 0,
}


def build_model():
    model = MUSIQ(
        patch_size=PATCH_SIZE, num_class=1,
        use_spherical_coords=True, use_face_emb=True, use_scale_emb=True,
        pretrained=False, num_faces=6, spatial_pos_grid_size=GRID_SIZE,
        num_heads=NUM_HEADS, dropout_rate=0.1, attention_dropout_rate=0.0,
        longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
    ).to(DEVICE)
    model.eval()
    return model


def load_fold_checkpoint(trained_on, fold):
    pattern = os.path.join(MODEL_DIR, f'{trained_on}_fold{fold}_epoch*')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No checkpoint matching {pattern}")
    path = files[0]
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict', ckpt)
    g_mean = float(ckpt.get('global_mean', 0.0))
    g_std  = float(ckpt.get('global_std',  1.0))
    return sd, g_mean, g_std, path


def read_scores(csv_path):
    names, scores = [], []
    with open(os.path.join(BASE, csv_path)) as f:
        for row in csv.DictReader(f):
            names.append(row['image_name'])
            scores.append(float(row['quality_score']))
    return names, scores


def resolve_paths(names, folder):
    folder = os.path.join(BASE, folder)
    paths = []
    for name in names:
        for ext in ('', '.jpg', '.png', '.jpeg', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'):
            p = os.path.join(folder, name + ext)
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(os.path.join(folder, name))
    return paths


def predict(model, paths, norm_scores, stereo, desc="Inference"):
    dataset = UnifiedODIQADataset(paths, norm_scores, hse_opts,
                                   augment=False, device='cpu', stereo=stereo)
    all_tta = []
    for yaw in TTA_ANGLES:
        dataset.yaw = yaw
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            num_workers=CPU_WORKERS, shuffle=False,
                            collate_fn=collate_fn)
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{desc} yaw={yaw}", leave=False):
                batch.pop('score')
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits, _ = model(**batch, return_aux=True)
                preds.extend(logits.view(-1).cpu().tolist())
        all_tta.append(preds)
    return np.mean(all_tta, axis=0)


# ── 1. Per-fold OOF predictions (diagonal) ──────────────────────────────

def evaluate_oof_save_preds(trained_on):
    """Evaluate OOF for each fold and save per-fold + combined prediction CSVs."""
    cfg = DATASET_CFG[trained_on]
    names, quality_scores = read_scores(cfg['csv'])
    paths = resolve_paths(names, cfg['dir'])
    stereo = cfg['stereo']
    image_scenes = get_scene_ids(trained_on, os.path.join(BASE, f"{trained_on}_scores.csv"))

    all_preds = []
    fold_metrics = []

    for fold in range(NUM_FOLDS):
        sd, g_mean, g_std, ckpt_path = load_fold_checkpoint(trained_on, fold)
        model = build_model()
        model.load_state_dict(sd)

        _, val_scenes = get_fold_split(trained_on, os.path.join(BASE, f"{trained_on}_scores.csv"),
                                       NUM_FOLDS, fold)
        val_idx = [i for i, s in enumerate(image_scenes) if s in val_scenes]
        val_paths = [paths[i] for i in val_idx]
        val_scores = [quality_scores[i] for i in val_idx]
        val_names = [names[i] for i in val_idx]
        norm_scores = [(s - g_mean) / g_std for s in val_scores]

        preds_z = predict(model, val_paths, norm_scores, stereo,
                          desc=f"{trained_on} fold{fold} OOF")
        preds_raw = [float(p * g_std + g_mean) for p in preds_z]
        pcc = pearsonr(preds_z, norm_scores)[0]
        srcc = spearmanr(preds_z, norm_scores)[0]
        fold_metrics.append((pcc, srcc))
        tqdm.write(f"  {trained_on} fold{fold}: PCC={pcc:.4f} SRCC={srcc:.4f} ({len(val_idx)} imgs)")

        fold_df = pd.DataFrame({
            'image_name': val_names,
            'fold': fold,
            'pred_z': [f'{p:.6f}' for p in preds_z],
            'pred_raw': [f'{p:.6f}' for p in preds_raw],
            'gt_quality': [f'{s:.6f}' for s in val_scores],
            'gt_norm': [f'{s:.6f}' for s in norm_scores],
        })

        # Save per-fold
        fold_csv = os.path.join(OUT, f'{trained_on}_fold{fold}_predictions.csv')
        fold_df.to_csv(fold_csv, index=False)

        all_preds.append(fold_df)

        del model
        torch.cuda.empty_cache()

    # Combined kfold predictions
    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(OUT, f'{trained_on}_kfold_predictions.csv'), index=False)
    tqdm.write(f"  Saved {trained_on}_kfold_predictions.csv ({len(combined)} rows)")

    pccs = [m[0] for m in fold_metrics]
    srccs = [m[1] for m in fold_metrics]
    tqdm.write(f"  {trained_on} OOF mean: PCC={np.mean(pccs):.4f} +/- {np.std(pccs):.4f}, SRCC={np.mean(srccs):.4f} +/- {np.std(srccs):.4f}")
    return fold_metrics


# ── 2. Cross-dataset evaluation (off-diagonal) ──────────────────────────

def evaluate_cross(trained_on, fold_models, evaluate_on_cfg):
    """Cross-dataset: average predictions across all 5 folds."""
    names, quality_scores = read_scores(evaluate_on_cfg['csv'])
    paths = resolve_paths(names, evaluate_on_cfg['dir'])
    stereo = evaluate_on_cfg['stereo']
    gt = np.array(quality_scores)

    all_preds_raw = []
    for fold, (model, g_mean, g_std) in enumerate(fold_models):
        norm_scores = [(s - g_mean) / g_std for s in quality_scores]
        preds_z = predict(model, paths, norm_scores, stereo,
                          desc=f"{trained_on}->{evaluate_on_cfg['csv'].replace('_scores.csv','')} fold{fold}")
        preds_raw = preds_z * g_std + g_mean
        all_preds_raw.append(preds_raw)
        torch.cuda.empty_cache()

    ensemble_preds = np.mean(all_preds_raw, axis=0)
    pcc = pearsonr(ensemble_preds, gt)[0]
    srcc = spearmanr(ensemble_preds, gt)[0]

    # Save per-image predictions
    eval_name = evaluate_on_cfg['csv'].replace('_scores.csv', '')
    cross_csv = os.path.join(OUT, f'cross_{trained_on}_on_{eval_name}.csv')
    pd.DataFrame({
        'image_name': names,
        'predicted_raw': [f'{p:.6f}' for p in ensemble_preds],
        'actual_raw': [f'{s:.6f}' for s in quality_scores],
    }).to_csv(cross_csv, index=False)

    return pcc, srcc


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross-only', action='store_true',
                        help='Skip Phase 1 OOF evaluation, only run cross-dataset matrix')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    datasets = sorted(DATASET_CFG.keys())

    if not args.cross_only:
        # Phase 1: Per-fold OOF predictions for all datasets (diagonal)
        print("\n" + "="*60)
        print("PHASE 1: Per-fold OOF predictions (diagonal)")
        print("="*60)
        for ds in datasets:
            print(f"\n--- {ds.upper()} ---")
            evaluate_oof_save_preds(ds)

    # Phase 2: Cross-dataset evaluation matrix
    print("\n" + "="*60)
    print("PHASE 2: Cross-dataset evaluation matrix")
    print("="*60)
    results = []

    for trained_on in datasets:
        print(f"\nLoading models trained on: {trained_on.upper()}")
        fold_models = []
        for fold in range(NUM_FOLDS):
            sd, g_mean, g_std, ckpt_path = load_fold_checkpoint(trained_on, fold)
            model = build_model()
            model.load_state_dict(sd)
            fold_models.append((model, g_mean, g_std))

        for evaluate_on in datasets:
            cfg = DATASET_CFG[evaluate_on]
            if trained_on == evaluate_on:
                # Use OOF metrics from Phase 1
                fold_csv = os.path.join(OUT, f'{trained_on}_kfold_predictions.csv')
                kdf = pd.read_csv(fold_csv)
                pr = kdf['pred_raw'].values.astype(float)
                gt = kdf['gt_quality'].values.astype(float)
                pcc = pearsonr(pr, gt)[0]
                srcc = spearmanr(pr, gt)[0]
            else:
                pcc, srcc = evaluate_cross(trained_on, fold_models, cfg)

            results.append({
                'trained_on': trained_on,
                'evaluate_on': evaluate_on,
                'PCC': round(float(pcc), 4),
                'SRCC': round(float(srcc), 4),
            })
            print(f"  {trained_on} -> {evaluate_on}:  PCC={pcc:.4f}  SRCC={srcc:.4f}")

        for m, _, _ in fold_models:
            del m
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'cross_validate_all_results.csv'), index=False)

    print("\n" + "="*60)
    print("PCC MATRIX")
    print("="*60)
    pcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    pcc_mat = pcc_mat[datasets].reindex(datasets)
    print(pcc_mat.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\n" + "="*60)
    print("SRCC MATRIX")
    print("="*60)
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[datasets].reindex(datasets)
    print(srcc_mat.to_string(float_format=lambda x: f'{x:.4f}'))

    print(f"\nAll results saved to {OUT}/")
    print("Done!")


if __name__ == '__main__':
    main()
