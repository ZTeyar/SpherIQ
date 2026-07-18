#!/usr/bin/env python3
"""
Cross-dataset evaluation using mean-of-per-fold-metrics instead of ensemble predictions.
For each trained_on→evaluate_on pair, computes PCC/SRCC per fold then averages.
"""
import os, sys, csv, glob, warnings, copy
warnings.filterwarnings("ignore", message="use_face_emb=True is set together")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_DIR)
from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.utils import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.path.join(REPO_DIR, 'best_models_100')
BATCH_SIZE = 1
CPU_WORKERS = 8
PATCH_SIZE = 32
GRID_SIZE = 10
NUM_HEADS = 4
TTA_ANGLES = [0]

DATASET_CFG = {
    'live': {'stereo': True,  'csv': 'live_scores.csv',  'dir': 'live'},
    'cviq': {'stereo': False, 'csv': 'cviq_scores.csv',  'dir': 'cviq'},
    'jufe': {'stereo': False, 'csv': 'jufe_scores.csv',  'dir': 'jufe'},
    'odi':  {'stereo': False, 'csv': 'odi_scores.csv',   'dir': 'odi'},
}

HSE_OPTS = {
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
    with open(os.path.join(REPO_DIR, csv_path)) as f:
        for row in csv.DictReader(f):
            names.append(row['image_name'])
            scores.append(float(row['quality_score']))
    return names, scores


def resolve_paths(names, folder):
    folder = os.path.join(REPO_DIR, folder)
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
    dataset = UnifiedODIQADataset(paths, norm_scores, HSE_OPTS,
                                   augment=False, device='cpu', stereo=stereo)
    all_tta = []
    for yaw in TTA_ANGLES:
        dataset.yaw = yaw
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            num_workers=CPU_WORKERS, shuffle=False,
                            collate_fn=collate_fn)
        preds = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
            for batch in tqdm(loader, desc=f"{desc} yaw={yaw}", leave=False):
                batch.pop('score')
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits, _ = model(**batch, return_aux=True)
                preds.extend(logits.view(-1).cpu().tolist())
        all_tta.append(preds)
    return np.mean(all_tta, axis=0)


def main():
    print(f"Device: {DEVICE}")
    datasets = sorted(DATASET_CFG.keys())
    results = []

    for trained_on in datasets:
        print(f"\n{'='*60}")
        print(f"Loading models trained on: {trained_on.upper()}")
        fold_models = []
        for fold in range(5):
            sd, g_mean, g_std, path = load_fold_checkpoint(trained_on, fold)
            model = build_model()
            model.load_state_dict(sd)
            fold_models.append((model, g_mean, g_std))
            print(f"  Fold {fold}: {os.path.basename(path)}")

        for evaluate_on in datasets:
            cfg = DATASET_CFG[evaluate_on]
            names, quality_scores = read_scores(cfg['csv'])
            paths = resolve_paths(names, cfg['dir'])
            stereo = cfg['stereo']
            gt = np.array(quality_scores)

            fold_pccs, fold_srccs = [], []
            for fold, (model, g_mean, g_std) in enumerate(fold_models):
                norm_scores = [(s - g_mean) / g_std for s in quality_scores]
                preds_z = predict(model, paths, norm_scores, stereo,
                                  desc=f"{trained_on}→{evaluate_on} fold{fold}")
                preds_raw = preds_z * g_std + g_mean
                pcc = pearsonr(preds_raw, gt)[0]
                srcc = spearmanr(preds_raw, gt)[0]
                fold_pccs.append(pcc)
                fold_srccs.append(srcc)
                torch.cuda.empty_cache()

            mean_pcc = float(np.mean(fold_pccs))
            mean_srcc = float(np.mean(fold_srccs))
            results.append({
                'trained_on': trained_on,
                'evaluate_on': evaluate_on,
                'PCC': round(mean_pcc, 4),
                'SRCC': round(mean_srcc, 4),
            })
            print(f"  {trained_on} → {evaluate_on}:  PCC={mean_pcc:.4f}  SRCC={mean_srcc:.4f}  "
                  f"(per-fold PCCs: {[f'{p:.4f}' for p in fold_pccs]})")

        for m, _, _ in fold_models:
            del m
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    out_path = os.path.join(REPO_DIR, 'output', 'cross_validate_per_fold_metrics.csv')
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("PCC MATRIX (mean of per-fold metrics)")
    print('='*60)
    pcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    pcc_mat = pcc_mat[datasets].reindex(datasets)
    print(pcc_mat.to_string(float_format=lambda x: f'{x:.4f}'))

    print(f"\n{'='*60}")
    print("SRCC MATRIX (mean of per-fold metrics)")
    print('='*60)
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[datasets].reindex(datasets)
    print(srcc_mat.to_string(float_format=lambda x: f'{x:.4f}'))

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
