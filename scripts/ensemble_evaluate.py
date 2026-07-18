import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from spheriq.evaluate import load_model, read_scores, resolve_paths, collate_fn
from spheriq.UnifiedDataset import UnifiedODIQADataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensemble_evaluate(
        dataset_folder,
        ground_truth_csv,
        num_folds=5,
        model_path_template="musiq_unified_trained_fold{fold}.pth",
        batch_size=1,
        patch_size=32,
        grid_size=10,
        num_heads=4,
        longer_side_lengths=None,
        output_csv="ensemble_results.csv",
        cpu_workers=4,
        tta_yaw_angles=None,
        use_face_emb=True,
        use_aux_head=False,
        stereo=False,
        max_seq_len_from_original_res=0,
):
    if tta_yaw_angles is None:
        tta_yaw_angles = [0]
    if longer_side_lengths is None:
        longer_side_lengths = [224, 384, 512]

    # 1. Load ground truth
    image_names, quality_scores, raw_scores, ds_means, ds_stds, ds_mins, ds_maxs, score_types = read_scores(ground_truth_csv)
    paths = resolve_paths(image_names, dataset_folder)
    
    if not paths:
        print("No images found for evaluation.")
        return

    # 2. Setup Dataset
    hse_opts = {
        'patch_size': patch_size,
        'patch_stride': patch_size,
        'hse_grid_size': grid_size,
        'longer_side_lengths': longer_side_lengths,
        'max_seq_len_from_original_res': max_seq_len_from_original_res,
    }
    
    # We'll normalize targets just for the dataloader consistency
    dataset = UnifiedODIQADataset(paths, quality_scores, hse_opts, augment=False, device='cpu', stereo=stereo)
    
    fold_predictions = [] # Will store final TTA-averaged predictions for each fold
    
    for fold in range(num_folds):
        model_path = model_path_template.format(fold=fold)
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Skipping fold {fold}.")
            continue
            
        print(f"\n" + "="*50)
        print(f"ENSEMBLE: RUNNING FOLD {fold} [{model_path}]")
        print("="*50)
        
        model, g_mean, g_std = load_model(
            model_path, 
            patch_size=patch_size, 
            device=DEVICE, 
            grid_size=grid_size,
            use_face_emb=use_face_emb,
            num_heads=num_heads,
            longer_side_lengths=longer_side_lengths,
            max_seq_len_from_original_res=max_seq_len_from_original_res
        )
        
        all_tta_preds = []
        for yaw in tta_yaw_angles:
            dataset.yaw = yaw
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=cpu_workers, shuffle=False, collate_fn=collate_fn)
            
            current_pass_preds = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Fold {fold} | Yaw {yaw}"):
                    batch.pop('score')
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    
                    logits, aux_pred = model(**batch, return_aux=True)
                    
                    if use_aux_head:
                        learned_w = torch.nn.functional.softmax(
                            torch.log(model.erp_face_weights + 1e-8),
                            dim=0,
                        )
                        preds = (aux_pred * learned_w.unsqueeze(0)).sum(dim=1, keepdim=True)
                        current_pass_preds.extend(preds.view(-1).cpu().tolist())
                    else:
                        current_pass_preds.extend(logits.view(-1).cpu().tolist())
            
            # Explicitly shut down workers to prevent "can only test a child process" errors
            del dataloader

            # Denormalize based on fold-specific global stats if they were used
            if g_mean is not None and g_std is not None:
                current_pass_preds = [p * g_std + g_mean for p in current_pass_preds]
            
            all_tta_preds.append(current_pass_preds)
            
        # Average across TTA passes for this model
        fold_final_preds = np.mean(all_tta_preds, axis=0)
        fold_predictions.append(fold_final_preds)
        
        del model
        torch.cuda.empty_cache()
        
    if not fold_predictions:
        print("No models were loaded successfully.")
        return
        
    # 3. Aggregate Ensemble Predictions (Average across folds)
    ensemble_preds = np.mean(fold_predictions, axis=0)
    
    # 4. Calculate Final Metrics
    # Note: we use raw quality_scores for final correlation calculation
    pcc = pearsonr(ensemble_preds, quality_scores)[0]
    srcc = spearmanr(ensemble_preds, quality_scores)[0]
    
    print("\n" + "="*30)
    print("FINAL ENSEMBLE SUMMARY")
    print(f"Models Ensembled: {len(fold_predictions)}")
    print(f"Ensemble PCC:     {pcc:.4f}")
    print(f"Ensemble SRCC:    {srcc:.4f}")
    print("="*30)
    
    # Per-scene analysis (matching evaluate.py logic)
    def get_scene(name):
        parts = name.replace(".jpg", "").replace(".png", "").split("_")
        if len(parts) >= 4:
            # ODI pattern: QUALITY_TRANSFERMODE_SCENETYPE_NUMBER
            return f"{parts[-2]}_{parts[-1]}"
        else:
            # LIVE3D pattern: SCENE_DISTORTION_LEVEL
            return parts[0]

    scenes = [get_scene(name) for name in image_names]
    unique_scenes = sorted(list(set(scenes)))
    if len(unique_scenes) > 1:
        print("\nPer-Scene Performance:")
        print(f"{'Scene':<15} | {'PCC':<8} | {'SRCC':<8} | {'N':<5}")
        print("-" * 45)
        for scene in unique_scenes:
            indices = [i for i, s in enumerate(scenes) if s == scene]
            s_preds = [ensemble_preds[i] for i in indices]
            s_actuals = [quality_scores[i] for i in indices]
            if len(s_preds) > 1:
                s_pcc = pearsonr(s_preds, s_actuals)[0]
                s_srcc = spearmanr(s_preds, s_actuals)[0]
                print(f"{scene:<15} | {s_pcc:<8.4f} | {s_srcc:<8.4f} | {len(s_preds):<5}")
        print("-" * 45)
    
    # 5. Save results
    out_df = pd.DataFrame({
        'image_name': image_names,
        'ensemble_predicted': ensemble_preds,
        'actual': quality_scores,
    })
    out_df.to_csv(output_csv, index=False)
    print(f"Ensemble results saved to {output_csv}")
    
    return ensemble_preds

if __name__ == "__main__":
    ensemble_evaluate(
        dataset_folder="live",
        ground_truth_csv="live_scores.csv",
        num_folds=5,
        tta_yaw_angles=[0, 180],
        stereo=True
    )