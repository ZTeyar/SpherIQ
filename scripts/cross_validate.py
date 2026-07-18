#!/usr/bin/env python3
import os, csv, warnings
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.utils import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_CFG = {
    'live': {'stereo': True,  'csv': 'live_scores.csv',  'dir': 'live'},
    'cviq': {'stereo': False, 'csv': 'cviq_scores.csv',  'dir': 'cviq'},
    'jufe': {'stereo': False, 'csv': 'jufe_scores.csv',  'dir': 'jufe'},
    'odi':  {'stereo': False, 'csv': 'odi_scores.csv',   'dir': 'odi'},
}

def load_model(model_path, grid_size=16):
    model = MUSIQ(
        patch_size=32, num_class=1,
        use_spherical_coords=True, use_face_emb=True, use_scale_emb=True,
        pretrained=False, num_faces=6, spatial_pos_grid_size=grid_size,
        num_heads=4, dropout_rate=0.2, attention_dropout_rate=0.1,
        longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
    ).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    model.eval()
    g_mean = ckpt.get('global_mean', 0.0)
    g_std  = ckpt.get('global_std',  1.0)
    return model, g_mean, g_std

def read_scores(csv_path):
    names, scores, raw, means, stds, mins, maxs, stypes = [], [], [], [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            names.append(row['image_name'])
            scores.append(float(row['quality_score']))
            raw.append(float(row.get('raw_score', row['quality_score'])))
            means.append(float(row.get('dataset_mean', 0.0)))
            stds.append(float(row.get('dataset_std', 1.0)))
            mins.append(float(row['dataset_min']) if 'dataset_min' in row else None)
            maxs.append(float(row['dataset_max']) if 'dataset_max' in row else None)
            stypes.append(row.get('score_type', 'DMOS'))
    return names, scores, raw, means, stds, mins, maxs, stypes

def resolve_paths(names, folder):
    paths = []
    for name in tqdm(names, desc="Resolving paths", leave=False):
        for ext in ('', '.jpg', '.png', '.jpeg', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'):
            p = os.path.join(folder, name + ext)
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(os.path.join(folder, name))
    return paths

def evaluate(trained_on, evaluate_on, tta_yaw_angles=None, batch_size=1,
             cpu_workers=4, grid_size=16):
    model_path = f'best_models/{trained_on}_best.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found')

    cfg = DATASET_CFG[evaluate_on]
    csv_path = cfg['csv']
    img_dir  = cfg['dir']
    stereo   = cfg['stereo']

    print(f'Evaluating [{trained_on}] model on [{evaluate_on}] dataset')
    model, g_mean, g_std = load_model(model_path, grid_size=grid_size)

    names, scores, raw, ds_means, ds_stds, ds_mins, ds_maxs, stypes = read_scores(csv_path)
    print(f'  Loaded {len(names)} samples from {csv_path}')
    paths = resolve_paths(names, img_dir)

    if abs(g_mean) > 1e-6 or abs(g_std - 1.0) > 1e-6:
        norm_scores = [(s - g_mean) / g_std for s in scores]
    else:
        norm_scores = scores

    hse_opts = {
        'patch_size': 32, 'patch_stride': 32, 'hse_grid_size': grid_size,
        'longer_side_lengths': [224, 384, 512], 'max_seq_len_from_original_res': 0,
    }
    dataset = UnifiedODIQADataset(paths, norm_scores, hse_opts,
                                   augment=False, device='cpu', stereo=stereo)

    if tta_yaw_angles is None:
        tta_yaw_angles = [0]

    all_tta = []
    for yaw in tta_yaw_angles:
        dataset.yaw = yaw
        loader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=cpu_workers, shuffle=False,
                           collate_fn=collate_fn)
        preds = []
        desc = f'  TTA yaw={yaw}' if yaw != 0 else '  Inference'
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch.pop('score')
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits, _ = model(**batch, return_aux=True)
                preds.extend(logits.view(-1).cpu().tolist())
        all_tta.append(preds)

    preds_z = np.mean(all_tta, axis=0)

    if len(preds_z) > 1:
        pcc  = pearsonr(preds_z,  norm_scores)[0]
        srcc = spearmanr(preds_z, norm_scores)[0]
    else:
        pcc, srcc = 0.0, 0.0

    print(f'  PCC={pcc:.4f}  SRCC={srcc:.4f}')
    return pcc, srcc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation')
    parser.add_argument('--trained_on', required=True, choices=list(DATASET_CFG),
                        help='Dataset the model was trained on')
    parser.add_argument('--evaluate_on', required=True, choices=list(DATASET_CFG),
                        help='Dataset to evaluate on')
    parser.add_argument('--tta', type=int, nargs='+', default=[0],
                        help='TTA yaw angles (e.g. 0 90 180 270)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grid_size', type=int, default=16,
                        help='Spatial grid size (must match training)')
    args = parser.parse_args()

    evaluate(args.trained_on, args.evaluate_on, tta_yaw_angles=args.tta,
             batch_size=args.batch_size, cpu_workers=args.workers,
             grid_size=args.grid_size)
