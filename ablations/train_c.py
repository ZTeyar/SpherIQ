#!/usr/bin/env python3
"""Variant C: VR-modified MUSIQ (spherical coords, 6 CLS, meta-transformer). Uses unified train_model()."""

import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from common import train_model, seed_everything, display_text, seed_worker

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.splits import get_fold_split, get_scene_ids
from spheriq.utils import collate_fn


def train_c(train_loader, val_loader, num_epochs=40, device='cuda',
            is_pretrained=True, save_prefix='variantC', continue_from=0):
    base_max_lr = 2e-5 if is_pretrained else 1e-4
    meta_max_lr = 1e-4 if is_pretrained else 4e-4

    model = MUSIQ(patch_size=32, num_class=1, spatial_pos_grid_size=10,
                  use_spherical_coords=True, use_face_emb=True, use_scale_emb=True,
                  dropout_rate=0.2, attention_dropout_rate=0.1, num_heads=4,
                  longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0,
                  pretrained=is_pretrained).to(device)

    backbone_params = (list(model.conv_root.parameters()) +
                       list(model.gn_root.parameters()) +
                       list(model.block1.parameters()) +
                       list(model.embedding.parameters()) +
                       list(model.transformer_encoder.parameters()))
    face_agg_params = (list(model.meta_transformer.parameters()) +
                       list(model.meta_head_mean.parameters()) +
                       list(model.meta_head_logvar.parameters()) +
                       [model.meta_agg_token] +
                       list(model.face_id_embed.parameters()) +
                       [model.nonpatch_attn_bias] +
                       [model.meta_query_bias_scale])
    head_params = list(model.head.parameters())

    param_groups = [
        {'params': backbone_params, 'lr': base_max_lr, 'weight_decay': 0.05},
        {'params': face_agg_params, 'lr': meta_max_lr, 'weight_decay': 0.01},
        {'params': head_params,     'lr': meta_max_lr, 'weight_decay': 0.01},
    ]

    model = train_model(
        model, train_loader, val_loader,
        param_groups=param_groups,
        freeze_params=backbone_params,
        freeze_epochs=2,
        num_epochs=num_epochs,
        device=device,
        save_prefix=save_prefix,
        continue_from=continue_from,
        val_tta_angles=[0],
        accumulation_steps=12,
    )
    return model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cviq')
    p.add_argument('--data-dir', default=os.getcwd())
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=3)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--save-prefix')
    p.add_argument('--continue-from', type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    prefix = args.save_prefix or f'variantC_{args.dataset}_fold{args.fold}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    display_text(f"Device={device}  Variant=C  Dataset={args.dataset}  Fold={args.fold}")

    # ── Load data ──────────────────────────────────────────────────────────
    score_file = os.path.join(os.path.abspath(args.data_dir), f'{args.dataset}_scores.csv')
    img_dir = os.path.join(os.path.abspath(args.data_dir), args.dataset)
    if not os.path.exists(score_file):
        display_text(f"Score file not found: {score_file}")
        return

    paths, scores = [], []
    scenes = get_scene_ids(args.dataset, score_file)
    with open(score_file) as f:
        for row in csv.DictReader(f):
            name = row['image_name']
            s = float(row['quality_score'])
            found = False
            for ext in ('.png', '.jpg', '.jpeg', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'):
                fp = os.path.join(img_dir, f'{name}{ext}')
                if os.path.exists(fp):
                    paths.append(fp)
                    found = True
                    break
            if not found:
                paths.append(os.path.join(img_dir, f'{name}.jpg'))
            scores.append(s)

    train_scenes, val_scenes = get_fold_split(args.dataset, score_file, args.num_folds, args.fold)
    display_text(f"Fold {args.fold+1}/{args.num_folds}: {len(train_scenes)} train, "
                 f"{len(val_scenes)} val scenes")

    train_idx = [i for i, s in enumerate(scenes) if s in train_scenes]
    val_idx = [i for i, s in enumerate(scenes) if s not in train_scenes]

    raw = [scores[i] for i in train_idx]
    gm = float(np.mean(raw))
    gs = float(np.std(raw))
    if gs < 1e-8:
        gs = 1.0
    display_text(f"Global z-score: mu={gm:.4f}, sigma={gs:.4f}")

    us = sorted(set(scenes))
    s2id = {s: i for i, s in enumerate(us)}
    opts = {'patch_size': 32, 'patch_stride': 32, 'hse_grid_size': 10,
            'longer_side_lengths': [224, 384, 512], 'max_seq_len_from_original_res': 0}

    td = UnifiedODIQADataset(
        [paths[i] for i in train_idx],
        [(scores[i] - gm) / gs for i in train_idx],
        opts, augment=True, device='cpu',
        scene_ids=[s2id[scenes[i]] for i in train_idx])
    vd = UnifiedODIQADataset(
        [paths[i] for i in val_idx],
        [(scores[i] - gm) / gs for i in val_idx],
        opts, augment=False, device='cpu',
        scene_ids=[s2id[scenes[i]] for i in val_idx])

    ct = ConcatDataset([td])
    cv = ConcatDataset([vd])
    ct.global_mean = gm
    ct.global_std = gs
    cv.global_mean = gm
    cv.global_std = gs

    tl = DataLoader(ct, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.cpu_workers, pin_memory=True,
                    collate_fn=collate_fn, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(args.seed))
    vl = DataLoader(cv, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.cpu_workers, pin_memory=True,
                    collate_fn=collate_fn, worker_init_fn=seed_worker,
                    generator=None)

    train_c(tl, vl, num_epochs=args.epochs, device=device,
            is_pretrained=not args.no_pretrained, save_prefix=prefix,
            continue_from=args.continue_from)


if __name__ == '__main__':
    main()
