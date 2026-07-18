#!/usr/bin/env python3
"""Variant B: vanilla pyiqa MUSIQ per cubemap face x 6, scores averaged. Uses unified train_model()."""

import sys
import os, math, copy

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import make_variant_b, load_ab_data, train_model, seed_everything, display_text


def train_b(train_loader, val_loader, num_epochs=40, device='cuda',
            is_pretrained=True, save_prefix='variantB', continue_from=0):
    bml = 2e-5 if is_pretrained else 1e-4
    mml = 1e-4 if is_pretrained else 4e-4

    model = make_variant_b(pretrained=is_pretrained).to(device)
    m = model.base
    bp = (list(m.conv_root.parameters()) + list(m.gn_root.parameters())
          + list(m.block1.parameters()) + list(m.embedding.parameters())
          + list(m.transformer_encoder.parameters()))
    hp = list(m.head.parameters())

    param_groups = [
        {'params': bp, 'lr': bml, 'weight_decay': 0.05},
        {'params': hp, 'lr': mml, 'weight_decay': 0.01},
    ]

    model = train_model(
        model, train_loader, val_loader,
        param_groups=param_groups,
        freeze_params=bp,
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
    prefix = args.save_prefix or f'variantB_{args.dataset}_fold{args.fold}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    display_text(f"Device={device}  Variant=B  Dataset={args.dataset}  Fold={args.fold}")

    tl, vl = load_ab_data('B', args.dataset, data_dir=args.data_dir,
                           num_folds=args.num_folds, fold_index=args.fold,
                           batch_size=args.batch_size, cpu_workers=args.cpu_workers,
                           seed=args.seed)
    train_b(tl, vl, num_epochs=args.epochs, device=device,
            is_pretrained=not args.no_pretrained, save_prefix=prefix,
            continue_from=args.continue_from)


if __name__ == '__main__':
    main()
