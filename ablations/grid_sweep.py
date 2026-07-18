#!/usr/bin/env python3
"""Grid-size sensitivity sweep: train LIVE Fold 4 with grid sizes {4,7,10,14}."""

import os, sys, csv, time, pandas as pd
import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from spheriq.train import train_on

os.chdir(REPO_DIR)

GRID_SIZES = [4, 7, 10, 14]
DATASET = 'live'
FOLD = 4
EPOCHS = 40

def run_grid_sweep():
    start = time.time()
    results = []
    for gs in GRID_SIZES:
        print(f"\n{'='*60}")
        print(f"GRID SIZE = {gs}  —  {DATASET} Fold {FOLD+1}")
        print(f"{'='*60}\n")
        t0 = time.time()
        train_on(
            datasets=[{'name': DATASET, 'stereo': True}],
            epochs=EPOCHS,
            patch_size=32,
            grid_size=gs,
            batch_size=2,
            cpu_workers=12,
            pretrained=True,
            val_tta_angles=[0],
            seed=42 + gs,
            freeze_base_epochs=10,
            pct_start=0.2,
            num_folds=5,
            fold_index=FOLD,
            artifact_aug_prob=0.0,
        )
        elapsed = (time.time() - t0) / 3600
        print(f"Grid size {gs} completed in {elapsed:.2f} hours.")

        log_file = f"{DATASET}_fold{FOLD}_training_logs.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            best = df.loc[df['Combined Val'].idxmax()]
            row = {
                'grid_size': gs,
                'pcc': best['Val PCC'],
                'srcc': best['Val SRCC'],
                'combined': best['Combined Val'],
                'epoch': int(best['Epoch']),
            }
        else:
            row = {'grid_size': gs, 'pcc': float('nan'), 'srcc': float('nan'),
                   'combined': float('nan'), 'epoch': -1, 'error': 'no log'}
        results.append(row)

        # Rename log/checkpoint to avoid overwrite by next grid size
        os.rename(log_file, f"{DATASET}_fold{FOLD}_grid{gs}_training_logs.csv")
        ckpt = f"musiq_unified_trained_fold{FOLD}.pth"
        if os.path.exists(ckpt):
            os.rename(ckpt, f"grid{gs}_{DATASET}_fold{FOLD}_best_checkpoint.pth")

    total = (time.time() - start) / 3600
    print(f"\n{'='*60}")
    print(f"GRID-SIZE SWEEP COMPLETE ({total:.2f} total hours)")
    print(f"{'='*60}")
    out = pd.DataFrame(results)
    out.to_csv('grid_sweep_results.csv', index=False)
    print(out.to_string(index=False))

if __name__ == '__main__':
    run_grid_sweep()
