from spheriq.utils import collate_fn
import gc
import os
import csv
import warnings
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from spheriq.musiq_arch import MUSIQ
from spheriq.UnifiedDataset import UnifiedODIQADataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_scores(ground_truth_scores_path):
    """
    Reads the processed CSV produced by prepare_scores.py.
    Returns image names, quality scores (for correlation), 
    and the per-image normalization parameters for denormalization.
    """
    if not os.path.exists(ground_truth_scores_path):
        print(f'Labels file [{ground_truth_scores_path}] not found.')
        return [], [], [], [], [], [], [], []

    image_names, quality_scores, raw_scores = [], [], []
    ds_means, ds_stds, ds_mins, ds_maxs, score_types = [], [], [], [], []

    with open(ground_truth_scores_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'quality_score' not in row:
                raise ValueError(
                    f"{ground_truth_scores_path} does not have a 'quality_score' column. "
                    "Run prepare_scores.process_labels() to generate the processed CSV first."
                )
            image_names.append(row['image_name'])
            quality_scores.append(float(row['quality_score']))
            raw_scores.append(float(row.get('raw_score', row['quality_score'])))
            ds_means.append(float(row.get('dataset_mean', 0.0)))
            ds_stds.append(float(row.get('dataset_std', 1.0)))
            
            # Support both old z-score and new min-max CSVs
            has_min = 'dataset_min' in row
            ds_mins.append(float(row['dataset_min']) if has_min else None)
            ds_maxs.append(float(row['dataset_max']) if has_min else None)
            
            score_types.append(row.get('score_type', 'DMOS'))

    return image_names, quality_scores, raw_scores, ds_means, ds_stds, ds_mins, ds_maxs, score_types


def resolve_paths(image_names, folder):
    paths = []
    for name in image_names:
        for ext in ('', '.jpg', '.png', '.jpeg', '.webp'):
            p = os.path.join(folder, name + ext)
            if os.path.exists(p):
                paths.append(p)
                break
        else:
            paths.append(os.path.join(folder, name))
    return paths


def load_model(model_path, patch_size=32, device="cuda", grid_size=10, 
               use_face_emb=True, num_heads=4, longer_side_lengths=None, 
               max_seq_len_from_original_res=0):
    
    import warnings
    warnings.filterwarnings("ignore", message="use_face_emb=True is set together")
    if longer_side_lengths is None:
        longer_side_lengths = [224, 384, 512]
        
    model = MUSIQ(
        patch_size=patch_size,
        num_class=1,
        use_spherical_coords=True,
        use_face_emb=use_face_emb,
        use_scale_emb=True,
        pretrained=False,
        num_faces=6,
        spatial_pos_grid_size=grid_size,
        num_heads=num_heads,
        dropout_rate=0.1,
        attention_dropout_rate=0.0,
        longer_side_lengths=longer_side_lengths,
        max_seq_len_from_original_res=max_seq_len_from_original_res,
    )
    model = model.to(device)

    if not os.path.exists(model_path):
        print(f'Model weights file [{model_path}] not found.\nDid you train the model?')
        return model, None, None
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
            state_dict = checkpoint['ema_state_dict']
            print(f"Loaded EMA weights from {model_path}")
        else:
            state_dict = (checkpoint['model_state_dict']
                          if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                          else checkpoint)
            print(f"Loaded raw weights from {model_path}")
            
        model.load_state_dict(state_dict)

    g_mean = checkpoint.get('global_mean') if isinstance(checkpoint, dict) else None
    g_std  = checkpoint.get('global_std')  if isinstance(checkpoint, dict) else None

    model.eval()
    return model, g_mean, g_std


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_model(
        model_path,
        dataset_folder,
        ground_truth_csv,
        batch_size=1,
        patch_size=32,
        grid_size=10,
        num_heads=4,
        longer_side_lengths=None,
        output_csv="evaluation_results.csv",
        cpu_workers=4,
        tta_yaw_angles=None,
        use_face_emb=True,
        use_aux_head=False,
        stereo=False,
        max_seq_len_from_original_res=0,
):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ['TORCH_LOG_LEVEL']      = 'ERROR'
    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    if tta_yaw_angles is None:
        # Default to [0] to match training validation (train.py defaults to [0])
        tta_yaw_angles = [0]

    if longer_side_lengths is None:
        longer_side_lengths = [224, 384, 512]

    hse_opts = {
        'patch_size':                    patch_size,
        'patch_stride':                  patch_size,
        'hse_grid_size':                 grid_size,
        'longer_side_lengths':           longer_side_lengths,
        'max_seq_len_from_original_res': max_seq_len_from_original_res,
    }

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

    image_names, quality_scores, raw_scores, ds_means, ds_stds, ds_mins, ds_maxs, score_types = read_scores(ground_truth_csv)
    paths = resolve_paths(image_names, dataset_folder)

    if not paths:
        print("No images found for evaluation. Check dataset_folder and ground_truth_csv.")
        return

    # ── Normalization (mirrors train_on() exactly) ───────────────────────────
    if g_mean is None or g_std is None:
        print("Checkpoint does not contain global normalization stats. Defaulting to no global normalization (mu=0.0, sigma=1.0).")
        g_mean, g_std = 0.0, 1.0

    if abs(g_mean) > 1e-6 or abs(g_std - 1.0) > 1e-6:
        norm_quality_scores = [(s - g_mean) / g_std for s in quality_scores]
        print(f"Applying global normalization mu={g_mean:.4f} sigma={g_std:.4f}")
    else:
        norm_quality_scores = quality_scores

    dataset = UnifiedODIQADataset(paths, norm_quality_scores, hse_opts, augment=False, device='cpu', stereo=stereo)
    
    # TTA logic
    tta_yaw_angles = tta_yaw_angles if tta_yaw_angles else [None]
    all_tta_preds = []

    print(f"Starting evaluation on {len(dataset)} images with TTA rotations: {tta_yaw_angles} ...")

    for pose in tta_yaw_angles:
        if isinstance(pose, (list, tuple)):
            if len(pose) == 3:
                yaw, pitch, roll = pose
            elif len(pose) == 2:
                yaw, pitch = pose
                roll = 0
            else:
                yaw = pose[0]
                pitch, roll = 0, 0
        else:
            yaw, pitch, roll = pose, 0, 0
            
        dataset.yaw = yaw
        dataset.pitch = pitch
        dataset.roll = roll
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cpu_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

        current_preds = []
        with torch.no_grad():
            pose_str = f" Yaw {yaw}" if yaw is not None else ""
            if pitch != 0 or roll != 0: pose_str += f" P{pitch} R{roll}"
            for batch in tqdm(dataloader, desc=pose_str.strip() if pose_str else "Processing"):
                batch.pop('score')
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                
                # Forward pass
                logits, aux_pred = model(**batch, return_aux=True)
                
                if use_aux_head:
                    # Mirror training validation aggregation: learned weighted average of face scores
                    learned_w = F.softmax(
                        torch.log(model.erp_face_weights + 1e-8),
                        dim=0,
                    )
                    # aux_pred shape: (B, 6)
                    preds = (aux_pred * learned_w.unsqueeze(0)).sum(dim=1, keepdim=True)
                    preds = preds.view(-1).cpu().tolist()
                else:
                    # Main prediction from Meta-Transformer (AGG token)
                    preds = logits.view(-1).cpu().tolist()
                    
                current_preds.extend(preds)
        
        all_tta_preds.append(current_preds)
        
        # Per-pass diagnostic
        if len(current_preds) > 1:
            pass_pcc = pearsonr(current_preds, norm_quality_scores)[0]
            print(f"  -> Pass PCC: {pass_pcc:.4f}")

    # Average across TTA passes
    predicted_quality_z = np.mean(all_tta_preds, axis=0).tolist()
    
    # Diagnostic: Check for constant inputs
    pred_std = np.std(predicted_quality_z)
    tgt_std  = np.std(norm_quality_scores)
    print(f"Predictions std: {pred_std:.6f} | Targets std: {tgt_std:.6f}")

    if len(predicted_quality_z) > 1:
        pcc  = pearsonr(predicted_quality_z,  norm_quality_scores)[0]
        srcc = spearmanr(predicted_quality_z, norm_quality_scores)[0]
        
        # Per-scene analysis for diagnostics
        def get_scene(name):
            parts = name.split('_')
            if len(parts) >= 4:
                # ODI pattern: QUALITY_TRANSFERMODE_SCENETYPE_NUMBER
                # Unique identifier is SCENETYPE + NUMBER (last two)
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
                s_preds = [predicted_quality_z[i] for i in indices]
                s_actuals = [norm_quality_scores[i] for i in indices]
                if len(s_preds) > 1:
                    s_pcc = pearsonr(s_preds, s_actuals)[0]
                    s_srcc = spearmanr(s_preds, s_actuals)[0]
                    print(f"{scene:<15} | {s_pcc:<8.4f} | {s_srcc:<8.4f} | {len(s_preds):<5}")
            print("-" * 45)
    else:
        pcc, srcc = 0.0, 0.0

    print(f"Spearman Rank Correlation (SRCC): {srcc:.4f}")
    print(f"Pearson Correlation        (PCC):  {pcc:.4f}")

    # 2. Denormalise to get the "true" ground truth scale (e.g. DMOS)
    predicted_raw = []
    for i, p_z in enumerate(predicted_quality_z):
        # Invert global z-score
        p_unscaled = p_z * g_std + g_mean
            
        # Invert per-dataset normalization
        if ds_mins[i] is not None and ds_maxs[i] is not None:
            range_val = ds_maxs[i] - ds_mins[i]
            if range_val < 1e-8: range_val = 1.0
            if score_types[i] == 'DMOS':
                p_raw = ds_maxs[i] - (p_unscaled * range_val)
            else:
                p_raw = ds_mins[i] + (p_unscaled * range_val)
        else:
            if score_types[i] == 'DMOS':
                p_raw = ds_means[i] - (p_unscaled * ds_stds[i])
            else:
                p_raw = ds_means[i] + (p_unscaled * ds_stds[i])
        predicted_raw.append(p_raw)

    out_df = pd.DataFrame({
        'image_name':            image_names,
        'predicted_z':           predicted_quality_z,
        'actual_z':              norm_quality_scores,
        'actual_quality_z':      quality_scores,
        'predicted_raw':         predicted_raw,
        'actual_raw':            raw_scores,
        'score_type':            score_types,
    })
    out_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MUSIQ Model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--csv", type=str, required=True, help="Path to ground truth scores CSV")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output results CSV")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size (must match training)")
    parser.add_argument("--grid_size", type=int, default=10, help="Spatial grid size (must match training)")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--workers", type=int, default=4, help="CPU workers for data loading")
    parser.add_argument("--tta", type=int, nargs="+", default=[0], help="Yaw angles for TTA (e.g. 0 90 180 270)")
    parser.add_argument("--aux", action="store_true", help="Use auxiliary face head for predictions")
    parser.add_argument("--stereo", action="store_true", help="Dataset contains stereoscopic ODS images")
    parser.add_argument("--max_seq_len", type=int, default=0, help="Max sequence length for original resolution patches")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        dataset_folder=args.dataset,
        ground_truth_csv=args.csv,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        grid_size=args.grid_size,
        num_heads=args.num_heads,
        output_csv=args.output,
        cpu_workers=args.workers,
        tta_yaw_angles=args.tta,
        use_aux_head=args.aux,
        stereo=args.stereo,
        max_seq_len_from_original_res=args.max_seq_len
    )