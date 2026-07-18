#!/usr/bin/env python3
"""
Full cross-dataset validation matrix for MUSIQ adaptation.

Evaluates all 4 datasets (live, cviq, odi, jufe) across all trained-on
combinations, producing a 4×4 table of PCC and SRCC.

For same-dataset (diagonal):  mean of per-fold OOF evaluations
For cross-dataset (off-diagonal):  average predictions across all 5 folds
"""
import glob, os, csv, warnings
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
MODEL_DIR = "best_models_100"

DATASET_CFG = {
    'live': {'stereo': True,  'csv': 'live_scores.csv',  'dir': 'live'},
    'cviq': {'stereo': False, 'csv': 'cviq_scores.csv',  'dir': 'cviq'},
    'jufe': {'stereo': False, 'csv': 'jufe_scores.csv',  'dir': 'jufe'},
    'odi':  {'stereo': False, 'csv': 'odi_scores.csv',   'dir': 'odi'},
}

NUM_FOLDS = 5
GRID_SIZE = 10
PATCH_SIZE = 32
NUM_HEADS = 4
BATCH_SIZE = 1
CPU_WORKERS = 12
TTA_ANGLES = [0]


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
    pattern = f"{MODEL_DIR}/{trained_on}_fold{fold}_epoch*"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint matching {pattern}")
    path = files[0]
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict', ckpt)
    g_mean = ckpt.get('global_mean', 0.0)
    g_std  = ckpt.get('global_std',  1.0)
    return sd, float(g_mean), float(g_std)


def load_fold_models(trained_on):
    models = []
    for fold in range(NUM_FOLDS):
        sd, g_mean, g_std = load_fold_checkpoint(trained_on, fold)
        model = build_model()
        model.load_state_dict(sd)
        models.append((model, g_mean, g_std))
    return models


def read_scores(csv_path):
    names, scores = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            names.append(row['image_name'])
            scores.append(float(row['quality_score']))
    return names, scores


def resolve_paths(names, folder):
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
    hse_opts = {
        'patch_size': PATCH_SIZE, 'patch_stride': PATCH_SIZE,
        'hse_grid_size': GRID_SIZE,
        'longer_side_lengths': [224, 384, 512],
        'max_seq_len_from_original_res': 0,
    }
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
            for batch in tqdm(loader, desc=f"{desc} yaw={yaw}", leave=True):
                batch.pop('score')
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits, _ = model(**batch, return_aux=True)
                preds.extend(logits.view(-1).cpu().tolist())
        del loader
        all_tta.append(preds)
    return np.mean(all_tta, axis=0)


def evaluate_oof(trained_on, fold_models, evaluate_on_cfg):
    """Mean of per-fold OOF evaluations."""
    names, quality_scores = read_scores(evaluate_on_cfg['csv'])
    paths = resolve_paths(names, evaluate_on_cfg['dir'])
    stereo = evaluate_on_cfg['stereo']
    image_scenes = get_scene_ids(trained_on, f"{trained_on}_scores.csv")

    fold_pccs, fold_srccs = [], []
    for fold, (model, g_mean, g_std) in enumerate(fold_models):
        _, val_scenes = get_fold_split(trained_on, f"{trained_on}_scores.csv",
                                       NUM_FOLDS, fold)
        val_idx = [i for i, s in enumerate(image_scenes) if s in val_scenes]

        val_paths = [paths[i] for i in val_idx]
        val_scores = [quality_scores[i] for i in val_idx]
        norm_scores = [(s - g_mean) / g_std for s in val_scores]

        preds_z = predict(model, val_paths, norm_scores, stereo,
                          desc=f"{trained_on} fold{fold} OOF")
        preds_raw = preds_z * g_std + g_mean

        pcc = pearsonr(preds_raw, val_scores)[0]
        srcc = spearmanr(preds_raw, val_scores)[0]
        fold_pccs.append(pcc)
        fold_srccs.append(srcc)
        tqdm.write(f"  Fold {fold}:  PCC={pcc:.4f}  SRCC={srcc:.4f}  ({len(val_idx)} imgs)")

        del model
        torch.cuda.empty_cache()

    return float(np.mean(fold_pccs)), float(np.mean(fold_srccs))


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
                          desc=f"{trained_on}→{evaluate_on_cfg['csv'].replace('_scores.csv','')} fold{fold}")
        preds_raw = preds_z * g_std + g_mean
        all_preds_raw.append(preds_raw)

        del model
        torch.cuda.empty_cache()

    ensemble_preds = np.mean(all_preds_raw, axis=0)
    pcc = pearsonr(ensemble_preds, gt)[0]
    srcc = spearmanr(ensemble_preds, gt)[0]
    return pcc, srcc


def main():
    print("Starting cross-validate-all evaluation...", flush=True)
    datasets = sorted(DATASET_CFG.keys())
    results = []

    for trained_on in datasets:
        print(f"\n{'='*60}")
        print(f"Loading models trained on: {trained_on.upper()}")
        print(f"{'='*60}")
        fold_models = load_fold_models(trained_on)

        for evaluate_on in datasets:
            cfg = DATASET_CFG[evaluate_on]

            if trained_on == evaluate_on:
                pcc, srcc = evaluate_oof(trained_on, fold_models, cfg)
            else:
                pcc, srcc = evaluate_cross(trained_on, fold_models, cfg)

            results.append({
                'trained_on': trained_on,
                'evaluate_on': evaluate_on,
                'PCC': round(pcc, 4),
                'SRCC': round(srcc, 4),
            })
            print(f"  RESULT: {trained_on} → {evaluate_on}:  PCC={pcc:.4f}  SRCC={srcc:.4f}")

    df = pd.DataFrame(results)

    print('\n' + '=' * 60)
    print('PCC MATRIX')
    print('=' * 60)
    pcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    pcc_mat = pcc_mat[datasets]
    pcc_mat = pcc_mat.reindex(datasets)
    print(pcc_mat.to_string(float_format=lambda x: f'{x:.4f}'))

    print('\n' + '=' * 60)
    print('SRCC MATRIX')
    print('=' * 60)
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[datasets]
    srcc_mat = srcc_mat.reindex(datasets)
    print(srcc_mat.to_string(float_format=lambda x: f'{x:.4f}'))

    out_csv = 'cross_validate_all_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved {out_csv}')


if __name__ == '__main__':
    main()
