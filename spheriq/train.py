import copy
import csv
import hashlib
import math
import os
import random
import time
import collections
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

from spheriq.UnifiedDataset import UnifiedODIQADataset
from spheriq.musiq_arch import MUSIQ
from spheriq.splits import get_fold_split, get_standard_split, get_scene_ids
from spheriq.utils import collate_fn
from spheriq.utils import display_text


# stable seed for shuffling data loaders
def get_stable_seed(input_string):
    hash_object = hashlib.md5(input_string.encode('utf-8'))
    return int(hash_object.hexdigest(), 16) % (2**32)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) of model weights
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Maintains an exponential moving average of model weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            
        exclude_buffers = {'erp_face_weights', 'meta_patch_bias'}
        model_buffers = dict(model.named_buffers())
        for name, ema_b in self.ema_model.named_buffers():
            if name not in model_buffers:
                continue
            model_b = model_buffers[name]
            if any(exc in name for exc in exclude_buffers):
                ema_b.data.copy_(model_b.data)
                continue
            if ema_b.dtype in (torch.float16, torch.float32, torch.float64):
                ema_b.data.mul_(self.decay).add_(model_b.data, alpha=1.0 - self.decay)
            else:
                ema_b.data.copy_(model_b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred.view(-1), target.view(-1), delta=self.delta)


class ERPWeightedGaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        erp_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(pred, tuple):
            mean, logvar = pred
            logvar = logvar.clamp(-2, 2)
        else:
            mean = pred
            logvar = torch.zeros_like(mean)

        if erp_weights is not None and mean.shape == erp_weights.shape:
            per_item = 0.5 * (logvar + (mean - target)**2 / torch.exp(logvar))
            return (per_item * erp_weights).sum() / erp_weights.sum().clamp(min=1e-8)
        else:
            return (0.5 * (logvar.view(-1) + (mean.view(-1) - target.view(-1))**2 / torch.exp(logvar.view(-1)))).mean()


class AdaptiveMarginRankingIQA(nn.Module):
    """
    Pairwise ranking loss with a margin that scales with the true quality gap.

    margin_i = alpha * |target_i - target_j|

    This means:
      - Pairs that are nearly identical in quality contribute ~zero margin → near-zero loss.
      - Pairs that are far apart in quality must be separated by a proportionally larger
        predicted gap, producing a stronger gradient signal.
      - alpha controls the aggressiveness. At alpha=0.5 with z-score targets in [-2,2],
        a gap of 1.0 z-unit imposes a 0.5-unit prediction margin.

    The filter `min_gap` removes pairs whose true gap is so small that the label
    difference is within measurement noise (typically ±0.05 z-units for small datasets).
    """

    def __init__(self, alpha: float = 0.5, min_gap: float = 0.05):
        super().__init__()
        self.alpha   = alpha
        self.min_gap = min_gap

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds   = preds.view(-1)
        targets = targets.view(-1)
        B = preds.size(0)
        if B < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        idx        = torch.combinations(torch.arange(B, device=preds.device), r=2)
        idx1, idx2 = idx[:, 0], idx[:, 1]

        p1, p2 = preds[idx1],   preds[idx2]
        t1, t2 = targets[idx1], targets[idx2]

        gap = t1 - t2
        y   = torch.sign(gap)

        # Filter: skip pairs too close to rank reliably
        valid = gap.abs() > self.min_gap
        if not valid.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        p1, p2, y, gap = p1[valid], p2[valid], y[valid], gap[valid]

        # Adaptive margin: larger gap → larger required prediction separation
        margin = self.alpha * gap.abs()                  # (n_valid,)

        # Equivalent to MarginRankingLoss but with per-pair margin:
        # loss = max(0, margin - y * (p1 - p2))
        loss = torch.clamp(margin - y * (p1 - p2), min=0.0)
        return loss.mean()

class SceneGroupedRankingLoss(nn.Module):
    """Only form pairs within the same scene group. Avoids cross-scene magnitude anchoring."""
    def __init__(self, alpha=0.3, min_gap=0.05):
        super().__init__()
        self.alpha = alpha
        self.min_gap = min_gap

    def forward(self, preds, targets, scene_ids):
        # scene_ids: (B,) integer tensor — scene index per sample in batch
        loss = torch.tensor(0.0, device=preds.device, requires_grad=True)
        count = 0
        unique_scenes = scene_ids.unique()
        for sid in unique_scenes:
            mask = (scene_ids == sid)
            if mask.sum() < 2:
                continue
            p, t = preds[mask].view(-1), targets[mask].view(-1)
            idx = torch.combinations(torch.arange(p.size(0), device=p.device), r=2)
            gap = t[idx[:,0]] - t[idx[:,1]]
            valid = gap.abs() > self.min_gap
            if not valid.any():
                continue
            margin = self.alpha * gap[valid].abs()
            y = torch.sign(gap[valid])
            raw = margin - y * (p[idx[valid,0]] - p[idx[valid,1]])
            loss = loss + F.relu(raw).mean()
            count += 1
        return loss / max(count, 1)

def calculate_metrics(predictions: torch.Tensor, scores: torch.Tensor) -> tuple[float, float]:
    p = predictions.cpu().numpy().flatten()
    s = scores.cpu().numpy().flatten()
    if np.std(p) < 1e-5 or np.std(s) < 1e-5:
        return 0.0, 0.0
    pcc  = pearsonr(p, s)[0]
    srcc = spearmanr(p, s)[0]
    return float(pcc), float(srcc)

def combined_metric(pcc: float, srcc: float, pcc_weight: float = 0.5) -> float:
    return pcc_weight * pcc + (1.0 - pcc_weight) * srcc

def save_checkpoint(epoch, model, optimizer, scheduler, pcc, srcc, path, global_mean=0.0, global_std=1.0, show=True, best=False, val_ema=None, ema: 'ModelEMA | None' = None):
    checkpoint = {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_pcc':             pcc,
        'best_srcc':            srcc,
        'best_metric':          combined_metric(pcc, srcc),
        'val_ema':              val_ema,
        'global_mean':          global_mean,
        'global_std':           global_std,
        'ema_state_dict':       ema.state_dict() if ema is not None else None,
    }
    torch.save(checkpoint, path)
    if show:
        tag = ' (best)' if best else ''
        ema_str = f"  EMA={val_ema:.4f}" if val_ema is not None else ""
        display_text(
            f"Checkpoint saved — epoch {epoch+1} | "
            f"PCC={pcc:.4f}  SRCC={srcc:.4f}{ema_str}{tag}"
        )

def save_epoch_logs(epoch, train_loss, val_loss,
                    train_pcc, train_srcc,
                    val_pcc,   val_srcc,
                    val_aux_pcc, val_aux_srcc,
                    val_ema,   epoch_duration, csv_filename):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Epoch',
                'Train Loss', 'Val Loss',
                'Train PCC',  'Val PCC',
                'Train SRCC', 'Val SRCC',
                'Val Aux PCC', 'Val Aux SRCC',
                'Combined Val', 'Val EMA (3-ep)', 'Training time (s)',
            ])
        writer.writerow([
            epoch,
            f'{train_loss:.4f}', f'{val_loss:.4f}',
            f'{train_pcc:.4f}',  f'{val_pcc:.4f}',
            f'{train_srcc:.4f}', f'{val_srcc:.4f}',
            f'{val_aux_pcc:.4f}', f'{val_aux_srcc:.4f}',
            f'{combined_metric(val_pcc, val_srcc):.4f}',
            f'{val_ema:.4f}',
            f'{int(epoch_duration)}',
        ])

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device='cuda', ema: 'ModelEMA | None' = None):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_state = (checkpoint['model_state_dict']
                     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                     else checkpoint)
        model.load_state_dict(raw_state)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError:
            pass
        if 'scheduler_state_dict' in checkpoint and scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except ValueError:
                pass

        if ema is not None and checkpoint.get('ema_state_dict') is not None:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', checkpoint.get('best_pcc', -1.0))
        val_ema = checkpoint.get('val_ema', None)
        pcc  = checkpoint.get('best_pcc',  best_metric)
        srcc = checkpoint.get('best_srcc', best_metric)
        g_mean = checkpoint.get('global_mean', 0.0)
        g_std  = checkpoint.get('global_std', 1.0)
        ema_str = f"  EMA={val_ema:.4f}" if val_ema is not None else ""
        display_text(
            f"Checkpoint loaded — resuming from epoch {epoch+2} | "
            f"PCC={pcc:.4f}  SRCC={srcc:.4f}{ema_str} | "
            f"Global stats: μ={g_mean:.4f} σ={g_std:.4f}"
        )
        return epoch + 1, best_metric, g_mean, g_std, val_ema, checkpoint
    return 0, -1.0, None, None, None, None

def train_musiq(model, train_loader, val_loader, num_epochs=10, device='cuda', datasets=None, continue_from=0, val_tta_angles=None, pct_start=0.3, freeze_base_epochs=2, is_pretrained=True, base_max_lr=None, meta_max_lr=None, ema_decay=0.995, num_folds=1, fold_index=0):
    if val_tta_angles is None:
        val_tta_angles = [0]
    if base_max_lr is None:
        base_max_lr = 2e-5 if is_pretrained else 1e-4
    if meta_max_lr is None:
        meta_max_lr = 1e-4 if is_pretrained else 4e-4
    backbone_params = list(model.conv_root.parameters()) + \
                      list(model.gn_root.parameters()) + \
                      list(model.block1.parameters()) + \
                      list(model.embedding.parameters()) + \
                      list(model.transformer_encoder.parameters())

    face_agg_params  = list(model.meta_transformer.parameters()) + \
                       list(model.meta_head_mean.parameters()) + \
                       list(model.meta_head_logvar.parameters()) + \
                       [model.meta_agg_token] + \
                       list(model.face_id_embed.parameters()) + \
                       [model.nonpatch_attn_bias] + \
                       [model.meta_query_bias_scale]

    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params,  'lr': base_max_lr,  'weight_decay': 0.05},
        {'params': face_agg_params,  'lr': meta_max_lr,  'weight_decay': 0.01},
        {'params': head_params,       'lr': meta_max_lr,  'weight_decay': 0.01},
    ])

    def _set_base_grad(requires: bool) -> None:
        for p in backbone_params:
            p.requires_grad_(requires)
            
    def _set_logvar_grad(requires: bool) -> None:
        for p in model.meta_head_logvar.parameters():
            p.requires_grad_(requires)
            
    _set_logvar_grad(False)
                
    if freeze_base_epochs > 0:
        _set_base_grad(False)
        display_text(f"Base model frozen for first {freeze_base_epochs} epochs.")
    accumulation_steps = 12
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[base_max_lr, meta_max_lr, meta_max_lr],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=pct_start
    )

    criterion_nll = ERPWeightedGaussianNLLLoss()
    # Targets are per-dataset min-max scores shifted by a global z-score.
    # The range is roughly [-2, 2]. temp=5.0 provides a soft ranking signal.
    criterion_rank = AdaptiveMarginRankingIQA(alpha=0.5, min_gap=0.05)
    criterion_scene_rank = SceneGroupedRankingLoss(alpha=0.3, min_gap=0.05)
    rank_weight    = 1.0   # increased to balance against Huber loss dominance
    scene_rank_weight = 0.3
    name_str = "_".join(datasets) if datasets else "musiq"
    if num_folds > 1:
        name_str = f"{name_str}_fold{fold_index}"
        
    weight_ema = ModelEMA(model, decay=ema_decay)
    start_epoch, best_metric, ckpt_mean, ckpt_std, val_ema, loaded_ckpt = load_checkpoint(
        model, optimizer, scheduler,
        f'{name_str}_checkpoints/checkpoint_{continue_from}.pth',
        device=device,
        ema=weight_ema,
    )
    best_model_state = copy.deepcopy(weight_ema.ema_model.state_dict())
    patience = 20
    epochs_no_improve = 0
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    scaler = torch.amp.GradScaler(device_type, enabled=(device_type == 'cuda'))
    
    if freeze_base_epochs > 0 and start_epoch >= freeze_base_epochs:
        _set_base_grad(True)
        display_text(f"Resuming from epoch {start_epoch + 1}: base model unfrozen.")

    if start_epoch >= freeze_base_epochs:
        _set_logvar_grad(True)
        display_text(f"Resuming from epoch {start_epoch + 1}: logvar head unfrozen.")

    for epoch in range(start_epoch, num_epochs):
        if freeze_base_epochs > 0 and epoch == freeze_base_epochs:
            _set_base_grad(True)
            display_text(f"Epoch {epoch + 1}: base model unfrozen.")
        if epoch == freeze_base_epochs:
            _set_logvar_grad(True)
            display_text(f'Epoch {epoch+1}: logvar head unfrozen.')
        epoch_start_time = time.time()
        model.train()
        train_nll_sum, train_rank_sum, train_steps = 0.0, 0.0, 0
        all_train_preds, all_train_scores = [], []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()
        accum_batches = []
        for batch_idx, batch in enumerate(train_bar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            scores      = batch.pop('score').view(-1, 1)
            scene_id    = batch.pop('scene_id', None)
            
            # Pass 1: Forward for Huber and detached predictions
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                predictions, aux_pred = model(**batch)
                
                if isinstance(predictions, tuple):
                    mean_pred, logvar_pred = predictions
                else:
                    mean_pred = predictions
                    
                loss_nll  = criterion_nll(predictions, scores, None)
                if aux_pred is not None and aux_pred.dim() == 2 and aux_pred.shape[1] == 6:
                    face_w = model.erp_face_weights.unsqueeze(0).expand_as(aux_pred)
                    loss_aux    = criterion_nll(aux_pred, scores.expand_as(aux_pred), face_w)
                else:
                    loss_aux    = criterion_nll(aux_pred, scores, None)
                # Backward NLL immediately to free the graph and save memory
                scaler.scale((loss_nll + 0.1 * loss_aux) / accumulation_steps).backward()
            
            # Detach predictions and require gradient for Two-Pass rank loss
            detached_pred = mean_pred.detach()
            detached_pred.requires_grad_(True)
            
            # Store micro-batch data for the second (rank) pass
            accum_batches.append({
                'batch': batch, 'scores': scores, 'pred': detached_pred,
                'rng_state': rng_state, 'cuda_rng_state': cuda_rng_state,
                'scene_id': scene_id
            })
            train_nll_sum += (loss_nll.item() + 0.1 * loss_aux.item())
            train_steps += 1
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Pass 2: Re-run forward passes for global Rank Loss
                all_preds_for_rank = [mb['pred'] for mb in accum_batches]
                all_scores_for_rank = [mb['scores'] for mb in accum_batches]
                
                loss_rank = criterion_rank(torch.cat(all_preds_for_rank), torch.cat(all_scores_for_rank))
                
                all_scenes_for_rank = [mb['scene_id'] for mb in accum_batches if mb['scene_id'] is not None]
                if all_scenes_for_rank:
                    loss_scene_rank = criterion_scene_rank(torch.cat(all_preds_for_rank), torch.cat(all_scores_for_rank), torch.cat(all_scenes_for_rank))
                else:
                    loss_scene_rank = torch.tensor(0.0, device=device)
                
                total_rank_loss = rank_weight * loss_rank + scene_rank_weight * loss_scene_rank
                
                # Scale to match one full Huber step magnitude: rank_weight * loss_rank
                scaler.scale(total_rank_loss).backward()
                
                # Update logging with rank contribution
                train_rank_sum += total_rank_loss.item()
                
                # Second pass: re-run the forward pass and backpropagate the rank loss gradients
                for mb in accum_batches:
                    grad = mb['pred'].grad
                    if grad is not None and grad.abs().sum() > 0:
                        # Restore RNG state to ensure identical Dropout masks as Pass 1
                        torch.set_rng_state(mb['rng_state'])
                        if mb['cuda_rng_state'] is not None:
                            torch.cuda.set_rng_state(mb['cuda_rng_state'])
                            
                        with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                            pred, _ = model(**mb['batch'], return_aux=False)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                        pred.backward(grad)

                
                has_grad = any(p.grad is not None for g in optimizer.param_groups for p in g['params'])
                
                if has_grad:
                    scaler.unscale_(optimizer)
                    
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                old_scale = scaler.get_scale()
                if has_grad:
                    scaler.step(optimizer)
                scaler.update()
                
                scale_reduced = scaler.get_scale() < old_scale
                if not scale_reduced:
                    if has_grad:
                        scheduler.step()
                optimizer.zero_grad()
                weight_ema.update(model)
                accum_batches.clear()

            all_train_preds.append(mean_pred.detach().cpu())
            all_train_scores.append(scores.cpu())
            true_loss = (train_nll_sum + train_rank_sum) / train_steps
            train_bar.set_postfix(loss=f'{true_loss:.4f}')
        train_pcc, train_srcc = calculate_metrics(torch.cat(all_train_preds), torch.cat(all_train_scores))
        eval_model = weight_ema.ema_model
        val_loss = 0.0
        val_loss_steps = 0
        all_tta_val_preds = []
        all_tta_val_aux = []
        all_val_scores = None
        val_tta_angles = val_tta_angles if val_tta_angles else [0]
        val_criterion  = HuberLoss(delta=0.5)
        for tta_idx, tta_pose in enumerate(val_tta_angles):
            yaw, pitch, roll = (tta_pose, 0, 0) if not isinstance(tta_pose, (list, tuple)) else (tta_pose[0], tta_pose[1], 0) if len(tta_pose) == 2 else tta_pose
            for ds in val_loader.dataset.datasets:
                ds.yaw, ds.pitch, ds.roll = yaw, pitch, roll
            current_val_loader = DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, collate_fn=val_loader.collate_fn, pin_memory=val_loader.pin_memory, generator=None)
            val_bar = tqdm(current_val_loader, desc=f"Epoch {epoch + 1} [Val]")
            
            tta_preds_list = []
            tta_aux_list = []
            tta_scores_list = []
            
            with torch.no_grad(), torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                for batch_idx, batch in enumerate(val_bar):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    scores = batch.pop('score').view(-1, 1)
                    batch.pop('scene_id', None)
                    predictions, aux_pred = eval_model(**batch, return_aux=True)
                    if aux_pred is not None and aux_pred.dim() == 2 and aux_pred.shape[1] == 6:
                        learned_w = F.softmax(
                            torch.log(eval_model.erp_face_weights + 1e-8),
                            dim=0,
                        )
                        aux_pred = (aux_pred * learned_w.unsqueeze(0)).sum(dim=1, keepdim=True)
                    step_loss = val_criterion(predictions, scores).item()
                    if tta_idx == 0:
                        val_loss += step_loss
                        val_loss_steps += 1
                    tta_preds_list.append(predictions.cpu())
                    tta_aux_list.append(aux_pred.cpu())
                    tta_scores_list.append(scores.cpu())
            
            current_tta_preds = torch.cat(tta_preds_list)
            current_tta_aux = torch.cat(tta_aux_list)
            current_tta_scores = torch.cat(tta_scores_list)
            
            all_tta_val_preds.append(current_tta_preds)
            all_tta_val_aux.append(current_tta_aux)
            if all_val_scores is None: all_val_scores = current_tta_scores
            
        final_val_preds = torch.stack(all_tta_val_preds).mean(dim=0)
        final_val_aux = torch.stack(all_tta_val_aux).mean(dim=0)
        
        val_pcc, val_srcc = calculate_metrics(final_val_preds, all_val_scores)
        val_aux_pcc, val_aux_srcc = calculate_metrics(final_val_aux, all_val_scores)
        
        val_combined = combined_metric(val_pcc, val_srcc)
        if val_ema is None: val_ema = val_combined
        else: val_ema = (1/3) * val_combined + (2/3) * val_ema
        
        display_text(f"Val Aux PCC: {val_aux_pcc:.4f} | Val Aux SRCC: {val_aux_srcc:.4f}")
        
        epoch_true_loss = (train_nll_sum + train_rank_sum) / max(1, train_steps)
        save_epoch_logs(epoch + 1, epoch_true_loss, val_loss / max(1, val_loss_steps), train_pcc, train_srcc, val_pcc, val_srcc, val_aux_pcc, val_aux_srcc, val_ema, time.time() - epoch_start_time, f'{name_str}_training_logs.csv')
        checkpoint_dir = f'{name_str}_checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        is_best = val_ema > best_metric
        save_checkpoint(epoch, model, optimizer, scheduler, val_pcc, val_srcc, os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}.pth'), global_mean=getattr(train_loader.dataset, 'global_mean', 0.0), global_std=getattr(train_loader.dataset, 'global_std', 1.0), best=is_best, val_ema=val_ema, ema=weight_ema)
        if is_best:
            best_metric = val_ema
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(weight_ema.ema_model.state_dict())
            torch.save({
                'model_state_dict': best_model_state,
                'global_mean': getattr(train_loader.dataset, 'global_mean', 0.0),
                'global_std':  getattr(train_loader.dataset, 'global_std',  1.0),
            }, f'{name_str}_best_checkpoint.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: break
    model.load_state_dict(best_model_state)
    model.eval()
    return model

def train_on(datasets=None, epochs=40, continue_from=0, patch_size=32, grid_size=10, batch_size=2, cpu_workers=8, pretrained=True, val_tta_angles=None, seed=42, pct_start=0.3, freeze_base_epochs=0, base_max_lr=None, meta_max_lr=None, ema_decay=None, num_folds=1, fold_index=0, artifact_aug_prob=0.3):
    import warnings
    warnings.filterwarnings("ignore", message="use_face_emb=True is set together")
    if val_tta_angles is None: val_tta_angles = [0, 90, 180, 270]
    seed_everything(seed)
    model = MUSIQ(patch_size=patch_size, num_class=1, spatial_pos_grid_size=grid_size, use_spherical_coords=True, use_face_emb=True, use_scale_emb=True, dropout_rate=0.2, attention_dropout_rate=0.1, num_heads=4, longer_side_lengths=[224, 384, 512], max_seq_len_from_original_res=0, pretrained=pretrained)
    hse_opts = {'patch_size': patch_size, 'patch_stride': patch_size, 'hse_grid_size': grid_size, 'longer_side_lengths': [224, 384, 512], 'max_seq_len_from_original_res': 0}
    all_raw_data = []
    ds_names = []
    for ds in datasets:
        name = ds['name']
        score_file = f'{name}_scores.csv'
        img_dir = f'{name}'
        paths, scores = [], []
        scenes = get_scene_ids(name, score_file)
        with open(score_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['image_name']
                score = float(row['quality_score'])
                for ext in ('.png', '.jpg', '.jpeg', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'):
                    if os.path.exists(os.path.join(img_dir, f"{img_name}{ext}")):
                        paths.append(os.path.join(img_dir, f"{img_name}{ext}"))
                        break
                else: paths.append(os.path.join(img_dir, f"{img_name}.jpg"))
                scores.append(score)

        if num_folds > 1:
            train_scenes, val_scenes = get_fold_split(name, score_file, num_folds, fold_index)
            display_text(f"Fold {fold_index+1}/{num_folds}: {len(train_scenes)} train, {len(val_scenes)} val scenes ({name})")
        else:
            train_scenes, val_scenes = get_standard_split(name, score_file)
            display_text(f"Standard split: {len(train_scenes)} train, {len(val_scenes)} val scenes ({name})")

        all_raw_data.append({'name': name, 'paths': paths, 'scores': scores, 'scenes': scenes, 'train_scenes': train_scenes, 'stereo': bool(ds.get('stereo', False))})
        ds_names.append(name)
    all_train_scores = [scores[i] for ds_data in all_raw_data for i, s in enumerate(ds_data['scenes']) if s in ds_data['train_scenes'] for scores in [ds_data['scores']]]
    gm, gs = float(np.mean(all_train_scores)), float(np.std(all_train_scores))
    if gs < 1e-8: gs = 1.0
    display_text(f"Global z-score: μ={gm:.4f}, σ={gs:.4f}")
    unique_scenes_global = list(set([s for ds_data in all_raw_data for s in ds_data['scenes']]))
    scene_to_id = {s: i for i, s in enumerate(unique_scenes_global)}

    train_data_list, val_data_list = [], []
    for ds_data in all_raw_data:
        train_idx = [i for i, s in enumerate(ds_data['scenes']) if s in ds_data['train_scenes']]
        val_idx = [i for i, s in enumerate(ds_data['scenes']) if s not in ds_data['train_scenes']]
        train_scene_ids = [scene_to_id[ds_data['scenes'][i]] for i in train_idx]
        val_scene_ids = [scene_to_id[ds_data['scenes'][i]] for i in val_idx]
        train_data_list.append(UnifiedODIQADataset([ds_data['paths'][i] for i in train_idx], [(ds_data['scores'][i] - gm) / gs for i in train_idx], hse_opts, augment=True, device='cpu', artifact_aug_prob=artifact_aug_prob, stereo=ds_data['stereo'], scene_ids=train_scene_ids))
        val_data_list.append(UnifiedODIQADataset([ds_data['paths'][i] for i in val_idx], [(ds_data['scores'][i] - gm) / gs for i in val_idx], hse_opts, augment=False, device='cpu', stereo=ds_data['stereo'], scene_ids=val_scene_ids))
    concat_train_ds, concat_val_ds = ConcatDataset(train_data_list), ConcatDataset(val_data_list)
    setattr(concat_train_ds, 'global_mean', gm)
    setattr(concat_train_ds, 'global_std', gs)
    if ema_decay is None:
        n_train = len(concat_train_ds)
        steps_per_epoch_est = max(1, math.ceil((n_train / batch_size) / 12))
        ema_decay = 0.5 ** (1.0 / (3.0 * steps_per_epoch_est))
        ema_decay = max(0.99, min(0.999, ema_decay))
        display_text(f"EMA decay auto-set to {ema_decay:.4f} (~3-epoch half-life for {n_train} training images)")
    train_loader = DataLoader(concat_train_ds, batch_size=batch_size, shuffle=True, num_workers=cpu_workers, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(concat_val_ds, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=True, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=None)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train_musiq(model, train_loader, val_loader, num_epochs=epochs, continue_from=continue_from, device='cuda' if torch.cuda.is_available() else 'cpu', datasets=ds_names, val_tta_angles=val_tta_angles, pct_start=pct_start, freeze_base_epochs=freeze_base_epochs, is_pretrained=pretrained, base_max_lr=base_max_lr, meta_max_lr=meta_max_lr, ema_decay=ema_decay, num_folds=num_folds, fold_index=fold_index)
    save_name = 'musiq_unified_trained.pth'
    if num_folds > 1:
        save_name = f'musiq_unified_trained_fold{fold_index}.pth'
    torch.save(trained_model.state_dict(), save_name)