#!/usr/bin/env python3
"""
Zero-shot evaluation of all 4 ablation variants on CVIQ.
Each variant's LIVE-fold-4 checkpoint is loaded and evaluated on CVIQ images.
"""
import os, sys, csv, copy, warnings
warnings.filterwarnings("ignore", message="use_face_emb=True is set together")
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from common import make_variant_a, make_variant_b, seed_worker
from common import ERPDataset, collate_erp, CubemapFaceWiseDataset, collate_faces

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.utils import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = REPO_DIR
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

BATCH_SIZE = 1
CPU_WORKERS = 8

HSE_OPTS = {
    'patch_size': 32, 'patch_stride': 32, 'hse_grid_size': 10,
    'longer_side_lengths': [224, 384, 512], 'max_seq_len_from_original_res': 0,
}

CHECKPOINTS = {
    'A': os.path.join(os.path.dirname(__file__), 'variantA_live_fold4_best_checkpoint.pth'),
    'B': os.path.join(os.path.dirname(__file__), 'variantB_live_fold4_best_checkpoint.pth'),
    'C': os.path.join(os.path.dirname(__file__), 'variantC_live_fold4_best_checkpoint.pth'),
    'D': os.path.join(REPO_DIR, 'best_models_100', 'live_fold4_epoch68_best.pth'),
    'FaceOnly': os.path.join(os.path.dirname(__file__), 'faceonly_live_fold4_best_checkpoint.pth'),
    'RoPEOnly': os.path.join(os.path.dirname(__file__), 'ropeonly_live_fold4_best_checkpoint.pth'),
}


def read_cviq_scores():
    path = os.path.join(DATA_DIR, 'cviq_scores.csv')
    names, scores = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row['image_name'])
            scores.append(float(row['quality_score']))
    return names, scores


def resolve_cviq_paths(names):
    folder = os.path.join(DATA_DIR, 'cviq')
    paths = []
    for name in names:
        for ext in ('', '.jpg', '.png', '.jpeg', '.webp', '.JPG', '.PNG', '.JPEG', '.WEBP'):
            p = os.path.join(folder, name + ext)
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(os.path.join(folder, name))
    return paths


def build_model_variant(variant):
    if variant == 'A':
        model = make_variant_a(pretrained=False).to(DEVICE)
    elif variant == 'B':
        model = make_variant_b(pretrained=False).to(DEVICE)
    elif variant == 'C':
        model = MUSIQ(patch_size=32, num_class=1, spatial_pos_grid_size=10,
                      use_spherical_coords=True, use_face_emb=True, use_scale_emb=True,
                      dropout_rate=0.2, attention_dropout_rate=0.1, num_heads=4,
                      longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
                      pretrained=False).to(DEVICE)
    elif variant == 'D':
        model = MUSIQ(patch_size=32, num_class=1, spatial_pos_grid_size=10,
                      use_spherical_coords=True, use_face_emb=True, use_scale_emb=True,
                      dropout_rate=0.2, attention_dropout_rate=0.1, num_heads=4,
                      longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
                      pretrained=False).to(DEVICE)
    elif variant == 'FaceOnly':
        model = MUSIQ(patch_size=32, num_class=1, spatial_pos_grid_size=10,
                      use_spherical_coords=False, use_face_emb=True, use_scale_emb=True,
                      dropout_rate=0.2, attention_dropout_rate=0.1, num_heads=4,
                      longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
                      pretrained=False).to(DEVICE)
    elif variant == 'RoPEOnly':
        model = MUSIQ(patch_size=32, num_class=1, spatial_pos_grid_size=10,
                      use_spherical_coords=True, use_face_emb=False, use_scale_emb=True,
                      dropout_rate=0.2, attention_dropout_rate=0.1, num_heads=4,
                      longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
                      pretrained=False).to(DEVICE)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model.eval()
    return model


def load_checkpoint(variant):
    ckpt_path = CHECKPOINTS[variant]
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict', ckpt)
    g_mean = float(ckpt.get('global_mean', 0.0))
    g_std  = float(ckpt.get('global_std', 1.0))
    return sd, g_mean, g_std, ckpt_path


def make_cviq_dataset(variant, norm_scores):
    paths = resolve_cviq_paths(names)
    if variant == 'A':
        ds = ERPDataset(paths, norm_scores, HSE_OPTS, augment=False)
        coll = collate_erp
    elif variant == 'B':
        ds = CubemapFaceWiseDataset(paths, norm_scores, HSE_OPTS, augment=False)
        coll = collate_faces
    else:
        ds = UnifiedODIQADataset(paths, norm_scores, HSE_OPTS,
                                  augment=False, device='cpu', stereo=False)
        coll = collate_fn
    return ds, coll


def evaluate(variant):
    print(f"\n{'='*60}")
    print(f"Variant {variant}")
    print(f"{'='*60}")

    sd, g_mean, g_std, ckpt_path = load_checkpoint(variant)
    model = build_model_variant(variant)
    model.load_state_dict(sd)
    print(f"  global_mean={g_mean:.4f}  global_std={g_std:.4f}")
    print(f"  checkpoint: {ckpt_path}")

    norm_scores = [(s - g_mean) / g_std for s in gt_scores]
    ds, coll = make_cviq_dataset(variant, norm_scores)

    loader = DataLoader(ConcatDataset([ds]), batch_size=BATCH_SIZE,
                        num_workers=CPU_WORKERS, shuffle=False,
                        collate_fn=coll)

    preds_z = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        for batch in tqdm(loader, desc=f"Variant {variant} on CVIQ", leave=True):
            batch.pop('score', None)
            batch.pop('scene_id', None)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits, _ = model(**batch, return_aux=True)
            pred = logits[0] if isinstance(logits, tuple) else logits
            preds_z.extend(pred.view(-1).cpu().tolist())

    preds_raw = np.array(preds_z) * g_std + g_mean
    gt_arr = np.array(gt_scores)

    pcc = pearsonr(preds_raw, gt_arr)[0]
    srcc = spearmanr(preds_raw, gt_arr)[0]
    print(f"  CVIQ PCC={pcc:.4f}  SRCC={srcc:.4f}")

    del model
    torch.cuda.empty_cache()
    return variant, pcc, srcc, preds_raw.tolist()


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    names, gt_scores = read_cviq_scores()
    print(f"CVIQ: {len(names)} images")

    results = []
    all_preds = {}
    for v in ('A', 'B', 'C', 'D', 'FaceOnly', 'RoPEOnly'):
        var, pcc, srcc, preds = evaluate(v)
        results.append({'variant': var, 'PCC': round(pcc, 4), 'SRCC': round(srcc, 4)})
        all_preds[var] = preds

    print(f"\n{'='*60}")
    print("CVIQ Zero-Shot Results")
    print(f"{'='*60}")
    for r in results:
        print(f"  Variant {r['variant']}:  PCC={r['PCC']:.4f}  SRCC={r['SRCC']:.4f}")

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, 'ablation_cviq_results.csv'), index=False)

    pred_df = pd.DataFrame({'image_name': names, 'gt': gt_scores})
    for var, preds in all_preds.items():
        pred_df[f'pred_{var}'] = preds
    pred_df.to_csv(os.path.join(OUT_DIR, 'ablation_cviq_predictions.csv'), index=False)

    print(f"\nSaved to {OUT_DIR}/ablation_cviq_results.csv")
