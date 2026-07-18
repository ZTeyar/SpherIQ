import os
import pandas as pd
import numpy as np
from spheriq.train import train_on
import time

def run_kfold_pipeline(only_fold=False, datasets=None, num_folds=5, epochs=40, continue_from=0, patch_size=32, grid_size=10, batch_size=2, cpu_workers=8, pretrained=True, val_tta_angles=None, seed=42, pct_start=0.3, freeze_base_epochs=0, base_max_lr=None, meta_max_lr=None, ema_decay=None, start_from_fold=0, artifact_aug_prob=0.3):
    if datasets is None:
        datasets = [{'name': 'live', 'stereo': True}]
    
    dataset_name = datasets[0]['name']
    
    start_time = time.time()
    
    print(f"Starting {num_folds}-Fold Cross-Validation for {dataset_name}...")
    
    for fold in range(start_from_fold, num_folds):
        print(f"\n" + "="*50)
        print(f"TRAINING FOLD {fold+1}/{num_folds}")
        print("="*50 + "\n")
        
        train_on(
            datasets=datasets,
            epochs=epochs,
            continue_from=continue_from,
            patch_size=patch_size,
            grid_size=grid_size,
            batch_size=batch_size,
            cpu_workers=cpu_workers,
            pretrained=pretrained,
            val_tta_angles=val_tta_angles,
            seed=seed,
            pct_start=pct_start,
            freeze_base_epochs=freeze_base_epochs,
            base_max_lr=base_max_lr,
            meta_max_lr=meta_max_lr,
            ema_decay=ema_decay,
            num_folds=num_folds,
            fold_index=fold,
            artifact_aug_prob=artifact_aug_prob,
        )
        continue_from=0
        if only_fold :
            break
        
    print(f"\n" + "="*50)
    print(f"AGGREGATING RESULTS")
    print("="*50 + "\n")
    
    results = []
    for fold in range(num_folds):
        log_file = f"{dataset_name}_fold{fold}_training_logs.csv"
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            if 'Combined Val' in df.columns:
                best_row = df.loc[df['Combined Val'].idxmax()]
                results.append({
                    'fold': fold,
                    'pcc': best_row['Val PCC'],
                    'srcc': best_row['Val SRCC'],
                    'combined': best_row['Combined Val']
                })
        else:
            print(f"Warning: Log file {log_file} not found.")
            
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print("\n" + "-"*30)
        print(f"FINAL K-FOLD SUMMARY ({num_folds} Folds)")
        print(f"Mean PCC:      {results_df['pcc'].mean():.4f} ± {results_df['pcc'].std():.4f}")
        print(f"Mean SRCC:     {results_df['srcc'].mean():.4f} ± {results_df['srcc'].std():.4f}")
        print(f"Mean Combined: {results_df['combined'].mean():.4f} ± {results_df['combined'].std():.4f}")
        print("-"*30)
    else:
        print("No results found to aggregate.")
        
    end_time = time.time()
    print(f"\nPipeline completed in {(end_time - start_time)/3600:.2f} hours.")

if __name__ == "__main__":
    # Configuration (mirrors your existing notebook settings)
    run_kfold_pipeline(
        datasets=[{'name': 'odi', 'stereo': False}],
        num_folds=5,
        epochs=100,
        patch_size=32,
        grid_size=10,
        batch_size=2,
        cpu_workers=12,
        pretrained=True,
        val_tta_angles=[0],
        start_from_fold=0,
        continue_from=0,
        artifact_aug_prob=0.0,
        freeze_base_epochs=10,
        pct_start=0.2
    )
