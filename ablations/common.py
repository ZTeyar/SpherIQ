#!/usr/bin/env python3
"""Shared code for all ablation variants."""

import argparse
import copy
import csv
import importlib.util
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm.auto import tqdm

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from spheriq.splits import get_fold_split, get_scene_ids
from spheriq.utils import display_text

from pyiqa.utils.registry import ARCH_REGISTRY
ARCH_REGISTRY._obj_map.pop('MUSIQ', None)

import pyiqa.archs.musiq_arch as pyiqa_musiq
from pyiqa.data.multiscale_trans_util import get_multiscale_patches

ARCH_REGISTRY._obj_map.pop('MUSIQ', None)

SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# Dataset: Variant A — ERP Direct
# ═════════════════════════════════════════════════════════════════════════════

class ERPDataset(Dataset):
    def __init__(self, image_paths, scores, hse_opts, augment=False,
                 scene_ids=None):
        self.image_paths = image_paths
        self.scores = scores
        self.scene_ids = scene_ids
        self.hse_opts = hse_opts
        self.augment = augment
        self.yaw = self.pitch = self.roll = None
        if augment:
            from torchvision import transforms
            self.jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as VF
        img_path = self.image_paths[idx]
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)
        img_pil = Image.open(img_path).convert('RGB')
        w, h = img_pil.size
        if w > 2048 or h > 1024:
            sf = min(2048 / w, 1024 / h)
            img_pil = img_pil.resize((int(w * sf), int(h * sf)), Image.Resampling.BICUBIC)
        if self.augment:
            img_pil = self.jitter(img_pil)
        img = VF.to_tensor(img_pil)
        img = (img - 0.5) * 2.0
        if self.yaw is not None and self.yaw != 0:
            shift = int(round(self.yaw / 360.0 * img.shape[2]))
            img = torch.roll(img, shifts=shift, dims=2)
        flat = get_multiscale_patches(img.unsqueeze(0), **self.hse_opts).squeeze(0)
        out = {'patches_flat': flat, 'score': score}
        if self.scene_ids is not None:
            out['scene_id'] = torch.tensor(self.scene_ids[idx])
        return out


def collate_erp(batch):
    max_len = max(item['patches_flat'].shape[0] for item in batch)
    dim = batch[0]['patches_flat'].shape[1]
    flats, scores, scenes = [], [], []
    has_scene = 'scene_id' in batch[0]
    for item in batch:
        n = item['patches_flat'].shape[0]
        pad = max_len - n
        if pad > 0:
            flats.append(torch.cat([item['patches_flat'], torch.zeros(pad, dim)], dim=0))
        else:
            flats.append(item['patches_flat'])
        scores.append(item['score'])
        if has_scene:
            scenes.append(item['scene_id'])
    out = {'x': torch.stack(flats), 'score': torch.stack(scores)}
    if has_scene:
        out['scene_id'] = torch.stack(scenes)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Dataset: Variant B — Cubemap faces, processed independently
# ═════════════════════════════════════════════════════════════════════════════

class CubemapFaceWiseDataset(Dataset):
    def __init__(self, image_paths, scores, hse_opts, augment=False,
                 device='cpu', artifact_aug_prob=0.3, stereo=False,
                 scene_ids=None):
        self.image_paths = image_paths
        self.scores = scores
        self.scene_ids = scene_ids
        self.hse_opts = hse_opts
        self.augment = augment
        self.artifact_aug_prob = artifact_aug_prob
        self.stereo = stereo
        self.yaw = self.pitch = self.roll = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import torchvision.transforms.functional as VF
        from PIL import Image
        from utils import get_cube_faces
        img_path = self.image_paths[idx]
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)
        img_pil = Image.open(img_path).convert('RGB')
        w, h = img_pil.size
        if w > 2048 or h > 1024:
            sf = min(2048 / w, 1024 / h)
            img_pil = img_pil.resize((int(w * sf), int(h * sf)), Image.Resampling.BICUBIC)
        img = VF.to_tensor(img_pil)
        img = (img - 0.5) * 2.0
        if self.yaw is not None and self.yaw != 0:
            shift = int(round(self.yaw / 360.0 * img.shape[2]))
            img = torch.roll(img, shifts=shift, dims=2)
        faces = get_cube_faces(img.unsqueeze(0)).squeeze(0)
        if self.augment:
            import torchvision.transforms as TVT
            noise = torch.randn(faces.shape) * 0.01
            faces = faces + noise
        faces_out = []
        for f in range(6):
            face = faces[f:f+1]
            flat = get_multiscale_patches(face, **self.hse_opts).squeeze(0)
            faces_out.append(flat)
        out = {'faces_flat': torch.stack(faces_out), 'score': score}
        if self.scene_ids is not None:
            out['scene_id'] = torch.tensor(self.scene_ids[idx])
        return out


def collate_faces(batch):
    max_n = max(item['faces_flat'].shape[1] for item in batch)
    dim = batch[0]['faces_flat'].shape[2]
    flats, scores, scenes = [], [], []
    has_scene = 'scene_id' in batch[0]
    for item in batch:
        n = item['faces_flat'].shape[1]
        pad = max_n - n
        if pad > 0:
            flats.append(torch.cat([item['faces_flat'], torch.zeros(6, pad, dim)], dim=1))
        else:
            flats.append(item['faces_flat'])
        scores.append(item['score'])
        if has_scene:
            scenes.append(item['scene_id'])
    out = {'x': torch.stack(flats), 'score': torch.stack(scores)}
    if has_scene:
        out['scene_id'] = torch.stack(scenes)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Model: Variant A — vanilla pyiqa MUSIQ wrapping patches
# ═════════════════════════════════════════════════════════════════════════════

class _VanillaMUSIQCore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def erp_face_weights(self):
        return torch.tensor([1.0], device=next(self.model.parameters()).device)

    def forward(self, x, return_aux=True, **kwargs):
        b, seq, dim = x.shape
        pos = x[:, :, -3].long()
        scale = x[:, :, -2].long()
        mask = x[:, :, -1].bool()
        pix = x[:, :, :-3]
        x_p = pix.reshape(-1, 3, self.model.patch_size, self.model.patch_size)
        x_p = self.model.conv_root(x_p); x_p = self.model.gn_root(x_p)
        x_p = self.model.root_pool(x_p); x_p = self.model.block1(x_p)
        x_p = x_p.permute(0, 2, 3, 1).reshape(b, seq, -1)
        x_p = self.model.embedding(x_p)
        x_p = self.model.transformer_encoder(x_p, pos, scale, mask)
        out = self.model.head(x_p[:, 0])
        if self.training:
            return (out, torch.zeros_like(out)), None
        return out, None


def make_variant_a(pretrained=True):
    m = pyiqa_musiq.MUSIQ(patch_size=32, num_class=1, hidden_size=384, mlp_dim=1152,
                           attention_dropout_rate=0.1, dropout_rate=0.2, num_heads=6,
                           num_layers=14, num_scales=3, spatial_pos_grid_size=10,
                           use_scale_emb=True, use_sinusoid_pos_emb=False,
                           pretrained=pretrained, longer_side_lengths=[224, 384, 512],
                           max_seq_len_from_original_res=0)
    if isinstance(m.head, nn.Sequential):
        m.head = nn.Linear(m.head[0].in_features, 1)
    return _VanillaMUSIQCore(m)


# ═════════════════════════════════════════════════════════════════════════════
# Model: Variant B — vanilla MUSIQ per face × 6, scores averaged
# ═════════════════════════════════════════════════════════════════════════════

class _VariantBModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def erp_face_weights(self):
        return torch.tensor([1.0], device=next(self.base.parameters()).device)

    def _cnn(self, pix):
        x_p = pix.reshape(-1, 3, self.base.patch_size, self.base.patch_size)
        x_p = self.base.conv_root(x_p); x_p = self.base.gn_root(x_p)
        x_p = self.base.root_pool(x_p); x_p = self.base.block1(x_p)
        x_p = x_p.permute(0, 2, 3, 1)
        x_p = x_p.reshape(pix.shape[0], pix.shape[1], -1)
        return self.base.embedding(x_p)

    def forward(self, x, return_aux=True, **kwargs):
        B, NF, N, dim = x.shape
        scores = []
        for i in range(NF):
            xi = x[:, i, :, :]
            pos = xi[:, :, -3].long(); scale = xi[:, :, -2].long()
            mask = xi[:, :, -1].bool(); pix = xi[:, :, :-3]
            if self.training:
                pix = pix.detach().requires_grad_(True)
                x_p = torch.utils.checkpoint.checkpoint(self._cnn, pix, use_reentrant=True)
            else:
                x_p = self._cnn(pix)
            x_p = self.base.transformer_encoder(x_p, pos, scale, mask)
            scores.append(self.base.head(x_p[:, 0]))
        out = torch.stack(scores, dim=1).mean(dim=1)
        if self.training:
            return (out, torch.zeros_like(out)), None
        return out, None


def make_variant_b(pretrained=True):
    m = pyiqa_musiq.MUSIQ(patch_size=32, num_class=1, hidden_size=384, mlp_dim=1152,
                           attention_dropout_rate=0.1, dropout_rate=0.2, num_heads=6,
                           num_layers=14, num_scales=3, spatial_pos_grid_size=10,
                           use_scale_emb=True, use_sinusoid_pos_emb=False,
                           pretrained=pretrained, longer_side_lengths=[224, 384, 512],
                           max_seq_len_from_original_res=0)
    in_features = m.head[0].in_features if isinstance(m.head, nn.Sequential) else m.head.in_features
    m.head = nn.Linear(in_features, 1)
    return _VariantBModel(m)


# ═════════════════════════════════════════════════════════════════════════════
# Metrics & Utilities
# ═════════════════════════════════════════════════════════════════════════════

def seed_everything(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, mp in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
        model_buf = dict(model.named_buffers())
        for name, eb in self.ema_model.named_buffers():
            if name not in model_buf:
                continue
            mb = model_buf[name]
            if any(exc in name for exc in ('erp_face_weights', 'meta_patch_bias')):
                eb.data.copy_(mb.data); continue
            if eb.dtype in (torch.float16, torch.float32, torch.float64):
                eb.data.mul_(self.decay).add_(mb.data, alpha=1.0 - self.decay)
            else:
                eb.data.copy_(mb.data)

    def state_dict(self):
        return self.ema_model.state_dict()


class AdaptiveMarginRankingIQA(nn.Module):
    def __init__(self, alpha=0.5, min_gap=0.05):
        super().__init__()
        self.alpha = alpha; self.min_gap = min_gap

    def forward(self, p, t):
        p = p.view(-1); t = t.view(-1)
        B = p.size(0)
        if B < 2:
            return torch.tensor(0.0, device=p.device, requires_grad=True)
        idx = torch.combinations(torch.arange(B, device=p.device), r=2)
        i1, i2 = idx[:, 0], idx[:, 1]
        gap = t[i1] - t[i2]
        valid = gap.abs() > self.min_gap
        if not valid.any():
            return torch.tensor(0.0, device=p.device, requires_grad=True)
        gap = gap[valid]
        margin = self.alpha * gap.abs()
        return torch.clamp(margin - torch.sign(gap) * (p[i1[valid]] - p[i2[valid]]), min=0.0).mean()


class SceneGroupedRankingLoss(nn.Module):
    """Only form pairs within the same scene group. Avoids cross-scene magnitude anchoring."""
    def __init__(self, alpha=0.3, min_gap=0.05):
        super().__init__()
        self.alpha = alpha
        self.min_gap = min_gap

    def forward(self, preds, targets, scene_ids):
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



def calc_metrics(p, t):
    p = p.cpu().numpy().flatten(); t = t.cpu().numpy().flatten()
    if np.std(p) < 1e-5 or np.std(t) < 1e-5:
        return 0.0, 0.0
    return float(pearsonr(p, t)[0]), float(spearmanr(p, t)[0])


# ═════════════════════════════════════════════════════════════════════════════
# Data loading (shared by A/B)
# ═════════════════════════════════════════════════════════════════════════════

def load_ab_data(variant, dataset_name, data_dir='.', num_folds=5, fold_index=3,
                 batch_size=2, cpu_workers=8, artifact_aug_prob=0.3, seed=None):
    if seed is None:
        seed = SEED
    score_file = os.path.join(data_dir, f'{dataset_name}_scores.csv')
    img_dir = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(img_dir):
        img_dir = data_dir
    paths, scores = [], []
    scenes = get_scene_ids(dataset_name, score_file)
    with open(score_file) as f:
        for row in csv.DictReader(f):
            name = row['image_name']
            s = float(row['quality_score'])
            found = False
            for ext in ('.png', '.jpg', '.jpeg', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'):
                fp = os.path.join(img_dir, f'{name}{ext}')
                if os.path.exists(fp):
                    paths.append(fp); found = True; break
            if not found:
                paths.append(os.path.join(img_dir, f'{name}.jpg'))
            scores.append(s)
    train_scenes, val_scenes = get_fold_split(dataset_name, score_file, num_folds, fold_index)
    train_idx = [i for i, s in enumerate(scenes) if s in train_scenes]
    val_idx = [i for i, s in enumerate(scenes) if s not in train_scenes]
    raw = [scores[i] for i in train_idx]
    gm = float(np.mean(raw)); gs = float(np.std(raw))
    if gs < 1e-8:
        gs = 1.0
    us = sorted(set(scenes))
    s2id = {s: i for i, s in enumerate(us)}
    opts = {'patch_size': 32, 'patch_stride': 32, 'hse_grid_size': 10,
            'longer_side_lengths': [224, 384, 512], 'max_seq_len_from_original_res': 0}
    if variant == 'A':
        td = ERPDataset([paths[i] for i in train_idx], [(scores[i]-gm)/gs for i in train_idx],
                        opts, augment=True, scene_ids=[s2id[scenes[i]] for i in train_idx])
        vd = ERPDataset([paths[i] for i in val_idx], [(scores[i]-gm)/gs for i in val_idx],
                        opts, augment=False, scene_ids=[s2id[scenes[i]] for i in val_idx])
        coll = collate_erp
    else:
        td = CubemapFaceWiseDataset([paths[i] for i in train_idx], [(scores[i]-gm)/gs for i in train_idx],
                                     opts, augment=True, device='cpu', artifact_aug_prob=artifact_aug_prob,
                                     stereo=False, scene_ids=[s2id[scenes[i]] for i in train_idx])
        vd = CubemapFaceWiseDataset([paths[i] for i in val_idx], [(scores[i]-gm)/gs for i in val_idx],
                                     opts, augment=False, device='cpu', stereo=False,
                                     scene_ids=[s2id[scenes[i]] for i in val_idx])
        coll = collate_faces
    ct = ConcatDataset([td]); cv = ConcatDataset([vd])
    ct.global_mean = gm; ct.global_std = gs
    cv.global_mean = gm; cv.global_std = gs
    tl = DataLoader(ct, batch_size=batch_size, shuffle=True, num_workers=cpu_workers,
                    pin_memory=True, collate_fn=coll, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(cv, batch_size=batch_size, shuffle=False, num_workers=cpu_workers,
                    pin_memory=True, collate_fn=coll, worker_init_fn=seed_worker, generator=None)
    return tl, vl


# ═════════════════════════════════════════════════════════════════════════════
# Unified training loop — shared by all 3 variants
# ═════════════════════════════════════════════════════════════════════════════

def train_model(
    model, train_loader, val_loader,
    param_groups,
    freeze_params=None,
    freeze_epochs=2,
    num_epochs=40,
    device='cuda',
    save_prefix='model',
    continue_from=0,
    val_tta_angles=None,
    accumulation_steps=12,
):
    if val_tta_angles is None:
        val_tta_angles = [0]

    opt = torch.optim.AdamW(param_groups)
    max_lrs = [g['lr'] for g in param_groups]
    spe = max(1, math.ceil(len(train_loader) / accumulation_steps))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, steps_per_epoch=spe, epochs=num_epochs, pct_start=0.3
    )

    criterion_rank = AdaptiveMarginRankingIQA(alpha=0.5, min_gap=0.05)
    criterion_scene_rank = SceneGroupedRankingLoss(alpha=0.3, min_gap=0.05)
    val_criterion = nn.HuberLoss(delta=0.5)

    n_train = len(train_loader.dataset)
    est = max(1, math.ceil((n_train / train_loader.batch_size) / 12))
    ed = max(0.99, min(0.999, 0.5 ** (1.0 / (3.0 * est))))
    wema = ModelEMA(model, decay=ed)

    best = -1.0
    best_state = copy.deepcopy(wema.ema_model.state_dict())
    patience = 20
    no_imp = 0
    dtype = 'cuda' if 'cuda' in str(device) else 'cpu'
    scaler = torch.amp.GradScaler(dtype, enabled=(dtype == 'cuda'))
    ckpt_dir = f'{save_prefix}_checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = f'{save_prefix}_training_logs.csv'

    start_epoch = 0
    if continue_from > 0:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{continue_from}.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            sched.load_state_dict(ckpt['scheduler_state_dict'])
            wema.ema_model.load_state_dict(ckpt['ema_state_dict'])
            best = ckpt.get('val_c', -1.0)
            no_imp = 0
            start_epoch = ckpt['epoch'] + 1
            display_text(f"Resumed from checkpoint_{continue_from}.pth (epoch {start_epoch})")
        else:
            display_text(f"Checkpoint not found: {ckpt_path}, starting from scratch")

    if freeze_params is not None and freeze_epochs > 0:
        for p in freeze_params:
            p.requires_grad_(False)

    for epoch in range(start_epoch, num_epochs):
        if freeze_params is not None and freeze_epochs > 0 and epoch == freeze_epochs:
            for p in freeze_params:
                p.requires_grad_(True)

        epoch_start = time.time()
        model.train()
        nll_sum = rank_sum = steps = 0.0
        all_preds, all_scores = [], []
        opt.zero_grad()
        accum = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for bidx, batch in enumerate(pbar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            scores = batch.pop('score').view(-1, 1)
            scene_id = batch.pop('scene_id', None)

            rng_state = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            with torch.amp.autocast(dtype, enabled=(dtype == 'cuda')):
                logits, _ = model(**batch, return_aux=True)
                mean_pred = logits[0] if isinstance(logits, tuple) else logits
                loss_nll = F.mse_loss(mean_pred, scores)
                scaler.scale(loss_nll / accumulation_steps).backward()

            d = mean_pred.detach()
            d.requires_grad_(True)
            accum.append({
                'batch': copy.deepcopy(batch), 'scores': scores, 'pred': d,
                'rng_state': rng_state, 'cuda_rng': cuda_rng,
                'scene_id': scene_id,
            })
            nll_sum += loss_nll.item()
            steps += 1

            if (bidx + 1) % accumulation_steps == 0 or (bidx + 1) == len(train_loader):
                rp = torch.cat([mb['pred'] for mb in accum])
                rs = torch.cat([mb['scores'] for mb in accum])
                lr = criterion_rank(rp, rs)
                lr_total = lr
                all_sids = [mb['scene_id'] for mb in accum if mb['scene_id'] is not None]
                if all_sids:
                    lr_scene = criterion_scene_rank(rp, rs, torch.cat(all_sids))
                    lr_total = lr_total + 0.3 * lr_scene
                scaler.scale(lr_total).backward()
                rank_sum += lr.item()

                for mb in accum:
                    g = mb['pred'].grad
                    if g is not None and g.abs().sum() > 0:
                        torch.set_rng_state(mb['rng_state'])
                        if mb['cuda_rng'] is not None:
                            torch.cuda.set_rng_state(mb['cuda_rng'])
                        with torch.amp.autocast(dtype, enabled=(dtype == 'cuda')):
                            p, _ = model(**mb['batch'], return_aux=False)
                            p = p[0] if isinstance(p, tuple) else p
                        p.backward(g)

                if any(p.grad is not None for g in opt.param_groups for p in g['params']):
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_sc = scaler.get_scale()
                if any(p.grad is not None for g in opt.param_groups for p in g['params']):
                    scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= old_sc:
                    sched.step()
                opt.zero_grad()
                wema.update(model)
                accum.clear()

            all_preds.append(mean_pred.detach().cpu())
            all_scores.append(scores.cpu())
            pbar.set_postfix(loss=f'{(nll_sum+rank_sum)/max(1,steps):.4f}')

        tp, ts = calc_metrics(torch.cat(all_preds), torch.cat(all_scores))

        em = wema.ema_model
        vl_sum = vl_steps = 0
        vp_all = []
        vs_all = None

        for tidx in val_tta_angles:
            for ds in val_loader.dataset.datasets:
                ds.yaw, ds.pitch, ds.roll = tidx, 0, 0
            vl = DataLoader(val_loader.dataset, batch_size=val_loader.batch_size,
                            num_workers=val_loader.num_workers, shuffle=False,
                            collate_fn=val_loader.collate_fn, pin_memory=val_loader.pin_memory)
            pts, sts = [], []
            with torch.no_grad(), torch.amp.autocast(dtype, enabled=(dtype == 'cuda')):
                for batch in tqdm(vl, desc=f"E {epoch+1} [Val TTA{tidx}]"):
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    sc = batch.pop('score', batch.pop('scores', None)).view(-1, 1)
                    batch.pop('scene_id', None)
                    lo, _ = em(**batch, return_aux=True)
                    pr = lo[0] if isinstance(lo, tuple) else lo
                    if tidx == val_tta_angles[0]:
                        vl_sum += val_criterion(pr, sc).item()
                        vl_steps += 1
                    pts.append(pr.cpu())
                    sts.append(sc.cpu())
            vp_all.append(torch.cat(pts))
            if vs_all is None:
                vs_all = torch.cat(sts)

        fp = torch.stack(vp_all).mean(0)
        vp, vs_ = calc_metrics(fp, vs_all)
        vc_metric = 0.5 * vp + 0.5 * vs_

        display_text(f"Ep {epoch+1} | L:{nll_sum/steps:.3f} R:{rank_sum/steps:.3f} | "
                     f"PCC {tp:.4f}/{vp:.4f} SRCC {ts:.4f}/{vs_:.4f}")

        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            if os.path.getsize(csv_path) == 0:
                w.writerow(['Epoch','TrainLoss','ValLoss','TrainPCC','ValPCC',
                            'TrainSRCC','ValSRCC','Combined','Time'])
            w.writerow([epoch+1, f'{nll_sum/max(1,steps):.4f}',
                        f'{vl_sum/max(1,vl_steps):.4f}', f'{tp:.4f}', f'{vp:.4f}',
                        f'{ts:.4f}', f'{vs_:.4f}', f'{vc_metric:.4f}',
                        f'{int(time.time()-epoch_start)}'])

        ib = vc_metric > best
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'val_pcc': vp, 'val_srcc': vs_, 'val_c': vc_metric,
            'ema_state_dict': wema.state_dict(),
            'global_mean': getattr(train_loader.dataset, 'global_mean', 0.0),
            'global_std': getattr(train_loader.dataset, 'global_std', 1.0),
        }, os.path.join(ckpt_dir, f'checkpoint_{epoch+1}.pth'))

        if ib:
            best = vc_metric
            no_imp = 0
            best_state = copy.deepcopy(wema.ema_model.state_dict())
            torch.save({
                'model_state_dict': best_state,
                'global_mean': getattr(train_loader.dataset, 'global_mean', 0.0),
                'global_std': getattr(train_loader.dataset, 'global_std', 1.0),
            }, f'{save_prefix}_best_checkpoint.pth')
            display_text(f"  → Best: {vc_metric:.4f}")
        else:
            no_imp += 1
            if no_imp >= patience:
                display_text(f"Early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model
