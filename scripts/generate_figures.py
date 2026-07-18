#!/usr/bin/env python3
"""
Generate all figures for the SpherIQ paper.
Reads data from repo root, outputs PNGs to imgs/.
"""
import os, glob, re, math, subprocess, json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from itertools import product, combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
from scipy.stats import pearsonr, spearmanr
import torch
from PIL import Image, ImageChops, ImageOps, ImageDraw, ImageFont

from spheriq.musiq_arch import MUSIQ

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def _scale_rc(scale):
    plt.rcParams.update({
        'font.size': 10 * scale,
        'axes.titlesize': 12 * scale,
        'axes.labelsize': 11 * scale,
        'xtick.labelsize': 9 * scale,
        'ytick.labelsize': 9 * scale,
        'legend.fontsize': 9 * scale,
    })

REPO = os.path.dirname(os.path.dirname(__file__))
BASE = REPO                                  # training logs only
PRED = REPO                                  # predictions + cross-val results
OUT = os.path.join(REPO, 'imgs')
os.makedirs(OUT, exist_ok=True)

DATASET_NAMES = ['cviq', 'jufe', 'live', 'odi']
DATASET_LABELS = ['CVIQ', 'JUFE', 'LIVE 3D', 'ODI']
COLORS = plt.cm.Set2(np.linspace(0, 1, 4))
NUM_FOLDS = 5

CROP_MARGIN = 10  # fixed pixels, not scaled by res_scale

def crop_margins(im_or_path, margin=CROP_MARGIN, thr=240):
    """Crop white margins from figure image, add fixed pixel border, return PIL Image.
    If im_or_path is a string path, saves the result back to the same file."""
    from PIL import ImageOps
    path = None
    if isinstance(im_or_path, str):
        path = im_or_path
        im = Image.open(path).convert('RGB')
    else:
        im = im_or_path.convert('RGB')
    arr = np.array(im)
    mask = (arr[..., 0] < thr) | (arr[..., 1] < thr) | (arr[..., 2] < thr)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        ys = np.where(rows)[0]
        xs = np.where(cols)[0]
        im = im.crop((int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1))
    result = ImageOps.expand(im, border=margin, fill='white')
    if path is not None:
        result.save(path, quality=95)
    return result


def load_training_logs(dataset):
    logs = {}
    for fold in range(NUM_FOLDS):
        path = os.path.join(BASE, f'{dataset}_fold{fold}_training_logs.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            epochs = df['Epoch'].values
            blocks = []
            start = 0
            for i in range(1, len(epochs)):
                if epochs[i] <= epochs[i - 1]:
                    blocks.append((start, i - 1))
                    start = i
            blocks.append((start, len(epochs) - 1))
            main = max(blocks, key=lambda b: epochs[b[1]])
            logs[fold] = df.iloc[main[0]:main[1] + 1].reset_index(drop=True)
    return logs


def get_best_per_fold(logs, metric='Val EMA (3-ep)'):
    best_epochs, best_vals, pccs, srccs = [], [], [], []
    for fold in sorted(logs.keys()):
        df = logs[fold]
        idx = df[metric].idxmax()
        best_epochs.append(int(df.loc[idx, 'Epoch']))
        best_vals.append(df.loc[idx, metric])
        pccs.append(df.loc[idx, 'Val PCC'])
        srccs.append(df.loc[idx, 'Val SRCC'])
    return best_epochs, best_vals, pccs, srccs


# ── 1. Train/Val Loss + PCC Curves (paper fig 7 + fig 8) ─────────────────
def _load_curves(dataset, res_scale=1.0):
    """Load and aggregate training curves across folds for a dataset.
    Returns (grid, mtl, stl, mvl, svl, mpcc, spcc, best_epoch) or None."""
    def load_runs(path):
        with open(path) as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        ncols = len(header)
        runs, cur = [], []
        for line in lines[1:]:
            s = line.strip()
            if not s:
                continue
            parts = s.split(',')
            try:
                ev = float(parts[0])
            except:
                continue
            if len(parts) >= 2 and abs(ev - 1.0) < 0.01 and len(cur) > 5:
                runs.append(pd.DataFrame(cur, columns=header[:len(cur[0])]))
                cur = []
            if len(parts) >= ncols:
                cur.append(parts[:ncols])
        if cur:
            runs.append(pd.DataFrame(cur, columns=header))
        return runs

    all_folds = {}
    for f in range(NUM_FOLDS):
        path = os.path.join(BASE, f'{dataset}_fold{f}_training_logs.csv')
        if not os.path.exists(path):
            continue
        runs = load_runs(path)
        best_run, best_val = None, -1
        for r in runs:
            vc = [c for c in r.columns if 'Val PCC' in c and 'Aux' not in c]
            if not vc:
                vc = [c for c in r.columns if 'Val PCC' in c]
            if vc:
                mx = r[vc[0]].astype(float).max()
                if mx > best_val:
                    best_val = mx
                    best_run = r
        if best_run is not None:
            all_folds[f] = best_run.apply(pd.to_numeric, errors='coerce')

    if not all_folds:
        return None

    max_ep = max(len(df) for df in all_folds.values())
    grid = np.arange(1, max_ep + 1)
    vpccs, vlosses, tlosses, cvals = [], [], [], []
    for f, df in all_folds.items():
        ep = df['Epoch'].values
        vc = [c for c in df.columns if 'Val PCC' in c and 'Aux' not in c]
        if not vc:
            continue
        vpccs.append(np.interp(grid, ep, df[vc[0]].values))
        vlosses.append(np.interp(grid, ep, df['Val Loss'].values))
        tlosses.append(np.interp(grid, ep, df['Train Loss'].values))
        cvals.append(np.interp(grid, ep, df['Combined Val'].values))

    vpccs = np.array(vpccs)
    vlosses = np.array(vlosses)
    tlosses = np.array(tlosses)
    cvals = np.array(cvals)
    mpcc, spcc = vpccs.mean(0), vpccs.std(0)
    mvl, svl = vlosses.mean(0), vlosses.std(0)
    mtl, stl = tlosses.mean(0), tlosses.std(0)
    mcv, scv = cvals.mean(0), cvals.std(0)
    best_idx = np.argmax(mpcc)
    best_epoch = grid[best_idx]
    return grid, mtl, stl, mvl, svl, mpcc, spcc, best_epoch, mcv, scv

def plot_loss_curves(res_scale=1.0):
    datasets = ['live', 'cviq']
    titles = ['LIVE 3D', 'CVIQ']
    colors = {'live': ('#2c7bb6', '#d7191c'), 'cviq': ('#41ab5d', '#fdae61')}

    all_data = {}
    for ds in datasets:
        d = _load_curves(ds, res_scale)
        if d is not None:
            all_data[ds] = d

    if not all_data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12 * res_scale, 7 * res_scale))
    for col, ds in enumerate(datasets):
        if ds not in all_data:
            continue
        grid, mtl, stl, mvl, svl, mpcc, spcc, best_epoch, mcv, scv = all_data[ds]
        tc, vc = colors[ds]
        label = titles[col]

        # Combined training loss (MSE + pairwise/rank)
        ax = axes[0, col]
        smtl = pd.Series(mtl).ewm(span=5).mean()
        sstl = pd.Series(stl).ewm(span=5).mean()
        ax.plot(grid, smtl, '-', color=tc, label='Train Loss (combined)', lw=1.2 * res_scale)
        ax.fill_between(grid, smtl - sstl, smtl + sstl, alpha=0.15, color=tc)
        ax.axvline(best_epoch, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=13 * res_scale)
        ax.set_ylabel('Loss', fontsize=13 * res_scale)
        ax.set_title(f'{label} — Training Loss (MSE + Rank)', fontsize=14 * res_scale, fontweight='bold')
        ax.tick_params(labelsize=11 * res_scale)
        ax.grid(True, alpha=0.3)

        # PCC
        ax = axes[1, col]
        ax.plot(grid, mpcc, '-', color=tc, label='Mean Val PCC', lw=1.2 * res_scale)
        ax.fill_between(grid, mpcc - spcc, mpcc + spcc, alpha=0.15, color=tc)
        ax.axvline(best_epoch, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=13 * res_scale)
        ax.set_ylabel('PCC', fontsize=13 * res_scale)
        ax.set_title(f'{label} — Validation PCC (5-fold CV)', fontsize=14 * res_scale, fontweight='bold')
        ax.legend(fontsize=11 * res_scale)
        ax.tick_params(labelsize=11 * res_scale)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    for ax in axes[:, 0]:
        ax.set_xlim(0, 80)
    for ax in axes[:, 1]:
        ax.set_xlim(0, 60)

    fig.savefig(os.path.join(OUT, 'fig7_training_loss.jpg'), dpi=50 * res_scale)
    plt.close(fig)
    print('  Saved fig7_training_loss.jpg')


# ── 2. LIVE out-of-fold scatter (paper fig) ──────────────────────────
def plot_live_scatter(res_scale=1.0):
    path = os.path.join(PRED, 'live_kfold_predictions.csv')
    if not os.path.exists(path):
        return
    kf = pd.read_csv(path)
    folds = sorted(kf['fold'].unique())
    fold_pccs, fold_srccs = [], []
    for fold in folds:
        sub = kf[kf['fold'] == fold]
        pz = sub['pred_z'].values.astype(float)
        gz = sub['gt_norm'].values.astype(float)
        fold_pccs.append(np.corrcoef(pz, gz)[0, 1])
        fold_srccs.append(spearmanr(pz, gz)[0])
    mean_pcc, std_pcc = np.mean(fold_pccs), np.std(fold_pccs)
    mean_srcc, std_srcc = np.mean(fold_srccs), np.std(fold_srccs)

    pz_all = kf['pred_z'].values.astype(float)
    gz_all = kf['gt_norm'].values.astype(float)

    fig, ax = plt.subplots(figsize=(5 * res_scale, 5 * res_scale))
    ax.scatter(gz_all, pz_all, alpha=0.4, s=15 * res_scale ** 2, c='#2c7bb6', edgecolors='none')
    m, b = np.polyfit(gz_all, pz_all, 1)
    xl = np.linspace(gz_all.min(), gz_all.max(), 100)
    ax.plot(xl, m * xl + b, 'r-', lw=1 * res_scale)
    ax.plot(xl, xl, 'k--', alpha=0.3, lw=0.8 * res_scale)
    ax.set_xlabel('Ground Truth DMOS (z-scored)', fontsize=12 * res_scale)
    ax.set_ylabel('Predicted DMOS (z-scored)', fontsize=12 * res_scale)
    ax.set_title('SpherIQ 5-Fold Cross-Validation on LIVE 3D', fontsize=13 * res_scale)
    ax.tick_params(labelsize=10 * res_scale)
    ax.text(0.05, 0.95, f'PCC = {mean_pcc:.4f} ± {std_pcc:.4f}\nSRCC = {mean_srcc:.4f} ± {std_srcc:.4f}',
            transform=ax.transAxes, fontsize=10 * res_scale, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig13_validation.jpg'), dpi=100 * res_scale)
    plt.close(fig)
    print('  Saved fig13_validation.jpg')


# ── 3. Ablation bar chart ───────────────────────────────────────────────
def plot_ablation(res_scale=1.0):
    abl_dir = os.path.join(BASE, '../paper/ablations')
    variants = [
        ('Variant A\nERP MUSIQ',  os.path.join(abl_dir, 'variantA_live_fold4_training_logs.csv'), '#d7191c'),
        ('Variant B\nCubemap x6', os.path.join(abl_dir, 'variantB_live_fold4_training_logs.csv'), '#fdae61'),
        ('Variant C\nVR MUSIQ',   os.path.join(abl_dir, 'variantC_live_fold4_training_logs.csv'), '#abd9e9'),
        ('Variant D\nSpherIQ',    os.path.join(BASE, 'live_fold4_training_logs.csv'),              '#2c7bb6'),
    ]
    labels, pccs, srccs, combined, epochs = [], [], [], [], []
    for name, path, _ in variants:
        df = pd.read_csv(path)
        # Normalize column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]
        # Find combined metric column
        comb_cols = [c for c in df.columns if 'combined' in c.lower() and 'val' not in c.lower()]
        comb_col = comb_cols[0] if comb_cols else None
        if comb_col is None or comb_col not in df.columns:
            comb_col = [c for c in df.columns if 'combined' in c.lower()][0]
        # Find Val PCC / Val SRCC columns (exclude Aux)
        pcc_col = [c for c in df.columns if 'val' in c.lower() and 'pcc' in c.lower() and 'aux' not in c.lower()][0]
        srcc_col = [c for c in df.columns if 'val' in c.lower() and 'srcc' in c.lower() and 'aux' not in c.lower()][0]
        idx = df[comb_col].astype(float).idxmax()
        labels.append(name)
        pccs.append(float(df.loc[idx, pcc_col]))
        srccs.append(float(df.loc[idx, srcc_col]))
        combined.append(float(df.loc[idx, comb_col]))
        epochs.append(int(df.loc[idx, 'Epoch']))

    fig, ax = plt.subplots(figsize=(7 * res_scale, 6 * res_scale))
    x = np.arange(len(labels))
    w = 0.28
    c1 = '#d62728'
    c2 = '#1f77b4'
    bars_pcc = ax.bar(x - w/2, pccs, w, label='PCC', color=c1, alpha=0.85, edgecolor='gray', lw=0.5 * res_scale)
    bars_srcc = ax.bar(x + w/2, srccs, w, label='SRCC', color=c2, alpha=0.85, edgecolor='gray', lw=0.5 * res_scale)
    for i in range(len(labels)):
        ax.text(i, max(pccs[i], srccs[i]), f'{combined[i]:.4f}', ha='center', va='bottom',
                fontsize=12 * res_scale, fontweight='bold')
    ax.set_ylabel('Validation Correlation', fontsize=14 * res_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10 * res_scale)
    ax.tick_params(axis='y', labelsize=13 * res_scale)
    ax.set_title('Ablation: Architecture Comparison on LIVE Fold 4', fontsize=14 * res_scale, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=12 * res_scale)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig10_ablation.jpg'), dpi=50 * res_scale)
    plt.close(fig)
    print('  Saved fig10_ablation.jpg')

    # Also print a text summary
    print()
    print('  Ablation Results (LIVE Fold 4):')
    print('  {:<20} {:>10} {:>8} {:>8} {:>6}'.format('Variant', 'Combined', 'PCC', 'SRCC', 'Epoch'))
    print('  ' + '-' * 52)
    for l, c, p, s, e in zip(labels, combined, pccs, srccs, epochs):
        ln = l.replace('\n', ' ')
        print('  {:<20} {:>10.4f} {:>8.4f} {:>8.4f} {:>6}'.format(ln, c, p, s, e))


# ── 3b. Geometric embedding ablation bar chart (fig geom_ablation) ─────
def plot_geom_ablation(res_scale=1.0):
    abl_dir = os.path.join(PRED, 'ablations')
    variants = [
        ('Plain Cubemap',  os.path.join(abl_dir, 'variantB_live_fold4_training_logs.csv')),
        ('+Face Embs',     os.path.join(abl_dir, 'faceonly_live_fold4_training_logs.csv')),
        ('+3D RoPE',       os.path.join(abl_dir, 'ropeonly_live_fold4_training_logs.csv')),
        ('Both',           os.path.join(abl_dir, 'variantC_live_fold4_training_logs.csv')),
    ]
    labels, pccs, srccs, combined, epochs = [], [], [], [], []
    for name, path in variants:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        comb_col = [c for c in df.columns if 'combined' in c.lower() and 'val' not in c.lower()][0]
        pcc_col = [c for c in df.columns if 'val' in c.lower() and 'pcc' in c.lower() and 'aux' not in c.lower()][0]
        srcc_col = [c for c in df.columns if 'val' in c.lower() and 'srcc' in c.lower() and 'aux' not in c.lower()][0]
        idx = df[comb_col].astype(float).idxmax()
        labels.append(name)
        pccs.append(float(df.loc[idx, pcc_col]))
        srccs.append(float(df.loc[idx, srcc_col]))
        combined.append(float(df.loc[idx, comb_col]))
        epochs.append(int(df.loc[idx, 'Epoch']))

    fig, ax = plt.subplots(figsize=(6 * res_scale, 4 * res_scale))
    x = np.arange(len(labels))
    w = 0.28
    ax.bar(x - w/2, pccs, w, label='PCC', color='#2c7bb6', alpha=0.85, edgecolor='gray', lw=0.5 * res_scale)
    ax.bar(x + w/2, srccs, w, label='SRCC', color='#d7191c', alpha=0.85, edgecolor='gray', lw=0.5 * res_scale)
    for i in range(len(labels)):
        ax.text(i, combined[i] + 0.02, f'{combined[i]:.4f}', ha='center', va='bottom',
                fontsize=9 * res_scale, fontweight='bold')
        ax.text(i, -0.08, f'ep{epochs[i]}', ha='center', va='top', fontsize=7 * res_scale, color='gray')
    ax.set_ylabel('Validation Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9 * res_scale)
    ax.set_title('Geometric Embedding Ablation (LIVE Fold 4)', fontsize=11 * res_scale, fontweight='bold')
    ax.set_ylim(0.5, 0.8)
    ax.legend(fontsize=9 * res_scale)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig11_geom_ablation.jpg'), dpi=200 * res_scale)
    plt.close(fig)
    print('  Saved fig11_geom_ablation.jpg')


# ── 4. Cross-dataset bar chart (paper fig) ──────────────────────────────
def plot_cross_dataset_bars(df, res_scale=1.0):
    fig, ax = plt.subplots(figsize=(7 * res_scale, 4 * res_scale))
    labels = [f"{r['trained_on']} -> {r['evaluate_on']}" for _, r in df.iterrows()]
    colors = ['#2c7bb6' if r['trained_on'] == r['evaluate_on'] else '#d7191c'
              for _, r in df.iterrows()]
    x = np.arange(len(labels))
    ax.bar(x, df['PCC'].values, color=colors, alpha=0.8, edgecolor='gray', lw=0.5 * res_scale)
    ax.set_ylabel('PCC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8 * res_scale)
    ax.set_title('Cross-Dataset Generalization')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend([Patch(color='#2c7bb6'), Patch(color='#d7191c')],
              ['In-distribution', 'Cross-dataset'], fontsize=9 * res_scale)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'cross_validation_bar.jpg'), dpi=200 * res_scale)
    plt.close(fig)
    print('  Saved cross_validation_bar.jpg')


# ── 5. Cross-Dataset Heatmap ────────────────────────────────────────────
def plot_heatmap(df, metric, fname, res_scale=1.0):
    fig, ax = plt.subplots(figsize=(5.5 * res_scale, 4.5 * res_scale))
    mat = df.pivot(index='trained_on', columns='evaluate_on', values=metric)
    mat = mat[DATASET_NAMES].reindex(DATASET_NAMES)
    vals = mat.values.astype(float)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect='equal')

    for i in range(4):
        for j in range(4):
            v = vals[i, j]
            ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                    fontsize=11 * res_scale, fontweight='bold',
                    color='white' if v > 0.65 else 'black')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(DATASET_LABELS, rotation=30, ha='right')
    ax.set_yticklabels(DATASET_LABELS)
    ax.set_xlabel('Evaluation Dataset')
    ax.set_ylabel('Training Dataset')

    for i in range(4):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1,
                                fill=False, edgecolor='red', lw=2.5 * res_scale))
    for i in range(4):
        ax.text(i, i, ' (OOF)', ha='left', va='center',
                fontsize=7 * res_scale, color='darkred', fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(metric)
    ax.set_title(f'Cross-Dataset {metric}', fontsize=12 * res_scale, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 6. Combined PCC/SRCC heatmaps ───────────────────────────────────────
def plot_dual_heatmap(df, fname, res_scale=1.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18 * res_scale, 7.5 * res_scale),
                                    gridspec_kw={'wspace': 0.35})
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for ax, metric, title in zip([ax1, ax2], ['PCC', 'SRCC'],
                                  ['(a) Pearson Correlation (PCC)',
                                   '(b) Spearman Correlation (SRCC)']):
        mat = df.pivot(index='trained_on', columns='evaluate_on', values=metric)
        mat = mat[DATASET_NAMES].reindex(DATASET_NAMES)
        vals = mat.values.astype(float)

        im = ax.imshow(vals, cmap=cmap, norm=norm, aspect='equal')
        for i in range(4):
            for j in range(4):
                v = vals[i, j]
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        fontsize=16 * res_scale, fontweight='bold',
                        color='white' if v > 0.65 else 'black')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(DATASET_LABELS, rotation=30, ha='right', fontsize=16 * res_scale)
        ax.set_yticklabels(DATASET_LABELS, fontsize=16 * res_scale)
        ax.set_xlabel('Evaluation Dataset', fontsize=16 * res_scale)
        ax.set_ylabel('Training Dataset', fontsize=16 * res_scale)
        ax.set_title(title, fontsize=17 * res_scale, fontweight='bold', pad=14)
        for i in range(4):
            ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1,
                                    fill=False, edgecolor='red', lw=2 * res_scale))

    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.02, pad=0.02)
    cbar.set_label('Correlation', fontsize=16 * res_scale)
    cbar.ax.tick_params(labelsize=14 * res_scale)
    fig.savefig(os.path.join(OUT, fname), bbox_inches='tight', pad_inches=0.6 * res_scale, dpi=50 * res_scale)
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 7. Per-Fold OOF Bar Chart ───────────────────────────────────────────
def plot_fold_oof_bars(fname, res_scale=1.0):
    datasets_with_folds = {}
    for ds in DATASET_NAMES:
        pred_files = sorted(glob.glob(os.path.join(PRED, f'{ds}_fold*_predictions.csv')))
        if pred_files:
            pccs, srccs = [], []
            for pf in pred_files:
                df = pd.read_csv(pf)
                p = df['pred_raw'].values.astype(float)
                g = df['gt_quality'].values.astype(float)
                pccs.append(pearsonr(p, g)[0])
                srccs.append(spearmanr(p, g)[0])
            datasets_with_folds[ds] = (pccs, srccs)

    if not datasets_with_folds:
        print('  No per-fold prediction CSVs found, skipping fold_oof_bars')
        return

    n = len(datasets_with_folds)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n * res_scale, 4 * res_scale))
    if n == 1:
        axes = [axes]

    cp = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    co = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))
    x = np.arange(5)

    for ax, (ds, (pccs, srccs)) in zip(axes, datasets_with_folds.items()):
        ax.bar(x - 0.175, pccs, 0.35, label='PCC', color=cp, edgecolor='k', lw=0.5 * res_scale)
        ax.bar(x + 0.175, srccs, 0.35, label='SRCC', color=co, edgecolor='k', lw=0.5 * res_scale)
        ax.axhline(np.mean(pccs), color='steelblue', ls='--', lw=1 * res_scale,
                   label=f'mu PCC={np.mean(pccs):.4f}')
        ax.axhline(np.mean(srccs), color='darkorange', ls=':', lw=1 * res_scale,
                   label=f'mu SRCC={np.mean(srccs):.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in range(5)])
        ax.set_ylim(0, 1)
        ax.set_title(f'{ds.upper()} Per-Fold OOF', fontsize=11 * res_scale, fontweight='bold')
        ax.set_ylabel('Correlation')
        ax.legend(fontsize=7 * res_scale, loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 8. Cross-Dataset Transfer Bars ──────────────────────────────────────
def plot_transfer_bars(df, fname, res_scale=1.0):
    pcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    pcc_mat = pcc_mat[DATASET_NAMES].reindex(DATASET_NAMES)
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[DATASET_NAMES].reindex(DATASET_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(12 * res_scale, 4.5 * res_scale))
    for ax, mat, metric in zip(axes, [pcc_mat, srcc_mat], ['PCC', 'SRCC']):
        x = np.arange(4)
        w = 0.18
        for i, (_, row) in enumerate(mat.iterrows()):
            offset = (i - 1.5) * w
            ax.bar(x + offset, row.values, w, label=row.name.upper(),
                   color=COLORS[i], edgecolor='k', lw=0.5 * res_scale)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_LABELS, rotation=20, ha='right')
        ax.set_ylabel(metric)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8 * res_scale, alpha=0.5)
        ax.set_title(f'{metric} by Training Dataset', fontsize=11 * res_scale, fontweight='bold')
        ax.legend(title='Train Set', fontsize=7 * res_scale, title_fontsize=8 * res_scale)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 9. OOF Scatter (LIVE + ODI) ─────────────────────────────────────────
def plot_scatters(fname, res_scale=1.0):
    kfold_files = {'LIVE 3D': 'live_kfold_predictions.csv',
                   'ODI': 'odi_kfold_predictions.csv'}
    existing = {k: os.path.join(PRED, v) for k, v in kfold_files.items()
                if os.path.exists(os.path.join(PRED, v))}
    if not existing:
        print('  No kfold prediction CSVs found, skipping scatters')
        return

    n = len(existing)
    fig, axes = plt.subplots(1, n, figsize=(5 * n * res_scale, 4.5 * res_scale))
    if n == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, existing.items()):
        df = pd.read_csv(path)
        oof = df.drop_duplicates(subset='image_name')
        pr = oof['pred_raw'].values.astype(float)
        gt = oof['gt_quality'].values.astype(float)
        pcc = pearsonr(pr, gt)[0]
        srcc = spearmanr(pr, gt)[0]

        ax.scatter(gt, pr, alpha=0.4, s=12 * res_scale ** 2, color=COLORS[0], edgecolors='none')
        lims = [min(gt.min(), pr.min()), max(gt.max(), pr.max())]
        ax.plot(lims, lims, 'r--', lw=1 * res_scale, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Ground Truth'); ax.set_ylabel('Predicted')
        ax.set_title(f'{label}\nPCC={pcc:.4f}  SRCC={srcc:.4f}',
                     fontsize=11 * res_scale, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 10. Training Curves (all datasets) ──────────────────────────────────
def plot_training_curves(fname, res_scale=1.0):
    fig, axes = plt.subplots(1, 3, figsize=(14 * res_scale, 4.2 * res_scale),
                              gridspec_kw={'wspace': 0.35})
    metrics = [('Val PCC', 'Validation PCC'),
               ('Val SRCC', 'Validation SRCC'),
               ('Val Loss', 'Validation Loss')]

    for ax, (col, label) in zip(axes, metrics):
        for ds, color in zip(DATASET_NAMES, COLORS):
            logs = load_training_logs(ds)
            if not logs:
                continue
            max_ep = max(df['Epoch'].max() for df in logs.values())
            epochs = np.arange(1, max_ep + 1)
            all_vals = []
            for fold in sorted(logs.keys()):
                vals = np.interp(epochs, logs[fold]['Epoch'].values, logs[fold][col].values)
                all_vals.append(vals)
            mean = np.mean(all_vals, axis=0)
            std = np.std(all_vals, axis=0)
            ax.plot(epochs, mean, color=color, label=ds.upper(), lw=1.5 * res_scale)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.legend(fontsize=7 * res_scale)
        ax.grid(True, alpha=0.3)
        if 'Loss' in label:
            ax.set_ylim(bottom=0)

    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 11. Best Performance Summary ────────────────────────────────────────
def plot_best_performance_summary(fname, res_scale=1.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9 * res_scale, 4 * res_scale), sharey=True)
    x = np.arange(len(DATASET_NAMES))
    for ax, metric, label in zip([ax1, ax2], ['Val PCC', 'Val SRCC'],
                                  ['Val PCC', 'Val SRCC']):
        means, stds = [], []
        for ds in DATASET_NAMES:
            logs = load_training_logs(ds)
            _, _, pccs, srccs = get_best_per_fold(logs)
            vals = pccs if metric == 'Val PCC' else srccs
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x, means, 0.3, yerr=stds, capsize=4 * res_scale, color=COLORS, edgecolor='k', lw=0.5 * res_scale)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f'{m:.4f}', ha='center', va='bottom',
                    fontsize=9 * res_scale, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_LABELS, rotation=20, ha='right')
        ax.set_ylabel(label)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    ax1.set_title('(a) Validation PCC', fontsize=11 * res_scale, fontweight='bold')
    ax2.set_title('(b) Validation SRCC', fontsize=11 * res_scale, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 12. Convergence Speed ───────────────────────────────────────────────
def plot_convergence(fname, res_scale=1.0):
    fig, ax = plt.subplots(figsize=(6 * res_scale, 4 * res_scale))
    x = np.arange(len(DATASET_NAMES))
    all_be = []
    for ds in DATASET_NAMES:
        logs = load_training_logs(ds)
        be, _, _, _ = get_best_per_fold(logs)
        all_be.append(be)
    means = [np.mean(be) for be in all_be]
    stds = [np.std(be) for be in all_be]

    ax.bar(x, means, 0.3, yerr=stds, capsize=4 * res_scale, color=COLORS, edgecolor='k', lw=0.5 * res_scale)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1, f'{m:.0f}', ha='center', va='bottom',
                fontsize=10 * res_scale, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, rotation=20, ha='right')
    ax.set_ylabel('Epoch to Best Validation')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')


# ── 13. Dashboard ───────────────────────────────────────────────────────
def plot_dashboard(df, fname, res_scale=1.0):
    fig = plt.figure(figsize=(16 * res_scale, 10 * res_scale))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # (a) PCC heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    mat = mat[DATASET_NAMES].reindex(DATASET_NAMES)
    vals = mat.values.astype(float)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im = ax1.imshow(vals, cmap=plt.cm.Blues, norm=norm, aspect='equal')
    for i in range(4):
        for j in range(4):
            v = vals[i, j]
            ax1.text(j, i, f'{v:.4f}', ha='center', va='center',
                     fontsize=9 * res_scale, fontweight='bold',
                     color='white' if v > 0.65 else 'black')
        ax1.add_patch(Rectangle((i-0.5, i-0.5), 1, 1,
                                fill=False, edgecolor='red', lw=2 * res_scale))
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xticklabels(DATASET_LABELS, rotation=30, ha='right', fontsize=8 * res_scale)
    ax1.set_yticklabels(DATASET_LABELS, fontsize=8 * res_scale)
    ax1.set_title('(a) PCC Cross-Dataset Matrix', fontsize=10 * res_scale, fontweight='bold')

    # (b) Training curves
    ax2 = fig.add_subplot(gs[0, 1:])
    for ds, color in zip(DATASET_NAMES, COLORS):
        logs = load_training_logs(ds)
        if not logs:
            continue
        max_ep = max(df['Epoch'].max() for df in logs.values())
        epochs = np.arange(1, max_ep + 1)
        all_vals = []
        for fold in sorted(logs.keys()):
            vals = np.interp(epochs, logs[fold]['Epoch'].values,
                             logs[fold]['Val EMA (3-ep)'].values)
            all_vals.append(vals)
        mean = np.mean(all_vals, axis=0)
        std = np.std(all_vals, axis=0)
        ax2.plot(epochs, mean, color=color, label=ds.upper(), lw=1.5 * res_scale)
        ax2.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.12)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val EMA')
    ax2.set_title('(b) Validation Training Curves', fontsize=10 * res_scale, fontweight='bold')
    ax2.legend(fontsize=8 * res_scale); ax2.grid(True, alpha=0.3)

    # (c) Best performance
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(DATASET_NAMES))
    mp, sp, ms, ss = [], [], [], []
    for ds in DATASET_NAMES:
        logs = load_training_logs(ds)
        _, _, pccs, srccs = get_best_per_fold(logs)
        mp.append(np.mean(pccs)); sp.append(np.std(pccs))
        ms.append(np.mean(srccs)); ss.append(np.std(srccs))
    ax3.bar(x - 0.15, mp, 0.3, yerr=sp, capsize=3 * res_scale, color='steelblue',
            edgecolor='k', lw=0.5 * res_scale, label='PCC')
    ax3.bar(x + 0.15, ms, 0.3, yerr=ss, capsize=3 * res_scale, color='darkorange',
            edgecolor='k', lw=0.5 * res_scale, label='SRCC')
    ax3.set_xticks(x); ax3.set_xticklabels(DATASET_LABELS, fontsize=8 * res_scale)
    ax3.set_ylabel('Best Val Correlation')
    ax3.set_title('(c) In-Dataset (OOF) Performance', fontsize=10 * res_scale, fontweight='bold')
    ax3.legend(fontsize=7 * res_scale); ax3.grid(True, axis='y', alpha=0.3)

    # (d) Cross-dataset SRCC
    ax4 = fig.add_subplot(gs[1, 1:])
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[DATASET_NAMES].reindex(DATASET_NAMES)
    x = np.arange(4); w = 0.18
    for i, (_, row) in enumerate(srcc_mat.iterrows()):
        offset = (i - 1.5) * w
        ax4.bar(x + offset, row.values, w, label=row.name.upper(),
                color=COLORS[i], edgecolor='k', lw=0.5 * res_scale)
    ax4.set_xticks(x); ax4.set_xticklabels(DATASET_LABELS, fontsize=8 * res_scale)
    ax4.set_ylabel('SRCC')
    ax4.set_title('(d) Cross-Dataset SRCC Breakdown', fontsize=10 * res_scale, fontweight='bold')
    ax4.legend(title='Train Set', fontsize=7 * res_scale, title_fontsize=8 * res_scale)
    ax4.grid(True, axis='y', alpha=0.3)

    # (e) Convergence
    ax5 = fig.add_subplot(gs[2, 0])
    all_be = []
    for ds in DATASET_NAMES:
        logs = load_training_logs(ds)
        be, _, _, _ = get_best_per_fold(logs)
        all_be.append(be)
    me = [np.mean(be) for be in all_be]
    se = [np.std(be) for be in all_be]
    ax5.bar(x, me, 0.4, yerr=se, capsize=3 * res_scale, color=COLORS, edgecolor='k', lw=0.5 * res_scale)
    for i, (m, s) in enumerate(zip(me, se)):
        ax5.text(i, m + s + 1, f'{m:.0f}', ha='center', fontsize=8 * res_scale, fontweight='bold')
    ax5.set_xticks(x); ax5.set_xticklabels(DATASET_LABELS, fontsize=8 * res_scale)
    ax5.set_ylabel('Epochs')
    ax5.set_title('(e) Convergence Speed', fontsize=10 * res_scale, fontweight='bold')
    ax5.grid(True, axis='y', alpha=0.3)

    # (f) LIVE scatter
    ax6 = fig.add_subplot(gs[2, 1])
    lkp = os.path.join(PRED, 'live_kfold_predictions.csv')
    if os.path.exists(lkp):
        ldf = pd.read_csv(lkp).drop_duplicates(subset='image_name')
        pr = ldf['pred_raw'].values.astype(float)
        gt = ldf['gt_quality'].values.astype(float)
        pcc = pearsonr(pr, gt)[0]; srcc = spearmanr(pr, gt)[0]
        ax6.scatter(gt, pr, alpha=0.3, s=8 * res_scale ** 2, color=COLORS[2], edgecolors='none')
        lims = [min(gt.min(), pr.min()), max(gt.max(), pr.max())]
        ax6.plot(lims, lims, 'r--', lw=1 * res_scale, alpha=0.5)
        ax6.set_xlim(lims); ax6.set_ylim(lims)
        ax6.set_xlabel('GT'); ax6.set_ylabel('Pred')
        ax6.set_title(f'(f) LIVE 3D OOF\nPCC={pcc:.4f} SRCC={srcc:.4f}',
                      fontsize=9 * res_scale, fontweight='bold')
        ax6.set_aspect('equal'); ax6.grid(True, alpha=0.3)

    # (g) ODI scatter
    ax7 = fig.add_subplot(gs[2, 2])
    okp = os.path.join(PRED, 'odi_kfold_predictions.csv')
    if os.path.exists(okp):
        odf = pd.read_csv(okp).drop_duplicates(subset='image_name')
        pr = odf['pred_raw'].values.astype(float)
        gt = odf['gt_quality'].values.astype(float)
        pcc = pearsonr(pr, gt)[0]; srcc = spearmanr(pr, gt)[0]
        ax7.scatter(gt, pr, alpha=0.3, s=8 * res_scale ** 2, color=COLORS[3], edgecolors='none')
        lims = [min(gt.min(), pr.min()), max(gt.max(), pr.max())]
        ax7.plot(lims, lims, 'r--', lw=1 * res_scale, alpha=0.5)
        ax7.set_xlim(lims); ax7.set_ylim(lims)
        ax7.set_xlabel('GT'); ax7.set_ylabel('Pred')
        ax7.set_title(f'(g) ODI OOF\nPCC={pcc:.4f} SRCC={srcc:.4f}',
                      fontsize=9 * res_scale, fontweight='bold')
        ax7.set_aspect('equal'); ax7.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f'  Saved {fname}')

# ── 12. Zero-shot scatter side-by-side (worst + best transfer) ─────────
def plot_zero_shot_scatter(res_scale=1.0):
    pairs = [
        ('cross_live_on_odi.csv', 'LIVE-trained → IQA-ODI', '#d7191c'),
        ('cross_odi_on_cviq.csv', 'IQA-ODI-trained → CVIQ', '#2c7bb6'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8 * res_scale, 3.5 * res_scale))
    for ax, (fname, label, color) in zip(axes, pairs):
        path = os.path.join(PRED, fname)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f'{fname} not found', ha='center', va='center', transform=ax.transAxes)
            continue
        df = pd.read_csv(path)
        pr = df['predicted_raw'].values.astype(float)
        gt = df['actual_raw'].values.astype(float)
        pcc = pearsonr(pr, gt)[0]
        srcc = spearmanr(pr, gt)[0]
        ax.scatter(gt, pr, alpha=0.4, s=12 * res_scale ** 2, color=color, edgecolors='none')
        lims = [min(gt.min(), pr.min()), max(gt.max(), pr.max())]
        ax.plot(lims, lims, 'k--', lw=1 * res_scale, alpha=0.6)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Ground Truth'); ax.set_ylabel('Predicted')
        ax.set_title(f'{label}\nPCC={pcc:.4f}  SRCC={srcc:.4f}', fontsize=10 * res_scale, fontweight='bold')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig14_transfer.jpg'), dpi=100 * res_scale)
    plt.close(fig)
    print('  Saved fig14_transfer.jpg (worst + best transfer)')


# ── 13. CVIQ ablation bar chart (all 4 variants zero-shot on CVIQ) ──────
def plot_cviq_ablation(res_scale=1.0):
    path = os.path.join(PRED, 'ablation_cviq_results.csv')
    if not os.path.exists(path):
        print('  ablation_cviq_results.csv not found, skipping cviq_ablation')
        return
    df = pd.read_csv(path)
    variants = df['variant'].values
    pccs = df['PCC'].values
    srccs = df['SRCC'].values

    fig, ax = plt.subplots(figsize=(5 * res_scale, 3.5 * res_scale))
    x = np.arange(len(variants))
    w = 0.35
    ax.bar(x - w/2, pccs, w, label='PCC', color='#2c7bb6', edgecolor='k', lw=0.5 * res_scale)
    ax.bar(x + w/2, srccs, w, label='SRCC', color='#d7191c', edgecolor='k', lw=0.5 * res_scale)
    for i in range(len(variants)):
        ax.text(i, pccs[i] + 0.01, f'{pccs[i]:.4f}', ha='center', fontsize=8 * res_scale)
        ax.text(i, srccs[i] + 0.01, f'{srccs[i]:.4f}', ha='center', fontsize=8 * res_scale)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Variant {v}' for v in variants])
    ax.set_ylabel('Correlation')
    ax.set_title('Zero-Shot CVIQ (LIVE Fold-4 Checkpoints)', fontsize=11 * res_scale, fontweight='bold')
    ax.legend(fontsize=9 * res_scale)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'cviq_ablation.jpg'), dpi=200 * res_scale)
    plt.close(fig)
    print('  Saved cviq_ablation.jpg')


# ── 3D Mapping (ported from sphviz.py) ────────────────────────────────
def draw_geodesic(ax, p1, p2, color='black', lw=2):
    t = np.linspace(0, 1, 20)
    omega = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    if omega < 1e-5:
        return
    arc = (
        (np.sin((1 - t) * omega) / np.sin(omega))[:, None] * p1
        + (np.sin(t * omega) / np.sin(omega))[:, None] * p2
    )
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color, linewidth=lw, linestyle='--')


def plot_3d_mapping(viz_grid=16, freq_scale=None, ckpt_path=None, use_window=True, res_scale=1.0):
    print('Generating fig9_3d_mapping (3D mapping figure)...')
    _render_scale = res_scale
    res_scale = res_scale * 0.5
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    })

    model = MUSIQ(
        num_faces=6,
        longer_side_lengths=[512],
        pretrained=False,
        use_spherical_coords=True,
        spatial_pos_grid_size=viz_grid,
        use_face_emb=False,
    )

    n_patches = viz_grid * viz_grid
    points_per_face = n_patches

    rows_per_face = torch.arange(viz_grid).repeat_interleave(viz_grid)
    cols_per_face = torch.arange(viz_grid).repeat(viz_grid)
    spatial_pos_per_face = rows_per_face * viz_grid + cols_per_face

    all_spatial_pos = spatial_pos_per_face.repeat(6).unsqueeze(0)
    face_ids = torch.arange(6).repeat_interleave(n_patches).unsqueeze(0)

    coords = model.transformer_encoder._get_rope_coords(
        all_spatial_pos, face_ids
    )[0, 6:].cpu().numpy()

    colors = ['red', 'deepskyblue', 'blue', 'darkorange', 'limegreen', 'darkviolet']
    idx_front = points_per_face * 4 + (viz_grid // 2 * viz_grid) + (viz_grid - 2)
    idx_right = points_per_face * 0 + (viz_grid // 2 * viz_grid) + 1

    def get_2d_coord(idx):
        face_idx = idx // points_per_face
        local_idx = idx % points_per_face
        r = local_idx // viz_grid
        c = local_idx % viz_grid
        face_grid = {
            0: (1, 2), 1: (1, 0), 2: (2, 1), 3: (0, 1), 4: (1, 1), 5: (1, 3),
        }
        row, col = face_grid[face_idx]
        spacing = 0.5
        start_x = col * (viz_grid + spacing)
        start_y = (2 - row) * (viz_grid + spacing)
        x = start_x + c + 0.5
        y = start_y + r + 0.5
        return x, y

    x_f_2d, y_f_2d = get_2d_coord(idx_front)
    x_r_2d, y_r_2d = get_2d_coord(idx_right)
    p1, p2 = coords[idx_front], coords[idx_right]

    # ── Part 1: 2D Sequence View (in-memory) ──────────────────────────────
    fig1 = plt.figure(figsize=(16 * res_scale, 2.5 * res_scale))
    ax2d = fig1.add_subplot(111)
    fig1.patch.set_facecolor('white')
    ax2d.set_title(
        r"Flattened 1D Patch Sequence (Topology-Agnostic)",
        fontsize=18 * res_scale, fontweight='bold', y=1.05,
    )

    samples_per_color = 10
    total_sampled = samples_per_color * 6
    cols_2d = 24
    patch_size = 1.0 * res_scale
    gap_x = 0.2 * res_scale
    gap_y = 0.8 * res_scale

    seq_colors = []
    for face_idx in range(6):
        seq_colors.extend([colors[face_idx]] * samples_per_color)

    for i in range(total_sampled):
        row = i // cols_2d
        col = i % cols_2d
        x = col * (patch_size + gap_x)
        y = -row * (patch_size + gap_y)
        rect = Rectangle(
            (x, y), patch_size, patch_size,
            facecolor=seq_colors[i], edgecolor='black', linewidth=1.2 * res_scale, alpha=0.9,
        )
        ax2d.add_patch(rect)
        if i == 5:
            ax2d.scatter(x + patch_size / 2, y + patch_size / 2,
                         color='darkred', s=150 * res_scale ** 2, edgecolors='white', zorder=100)
            ax2d.text(x + patch_size / 2, y - 0.5 * res_scale, r'$p_j$',
                      color='black', fontsize=18 * res_scale, ha='center', fontweight='bold')
        if i == 45:
            ax2d.scatter(x + patch_size / 2, y + patch_size / 2,
                         color='red', s=150 * res_scale ** 2, edgecolors='white', zorder=100)
            ax2d.text(x + patch_size / 2, y - 0.5 * res_scale, r'$p_i$',
                      color='black', fontsize=18 * res_scale, ha='center', fontweight='bold')

    ax2d.set_xlim(-1 * res_scale, cols_2d * (patch_size + gap_x) + 1 * res_scale)
    ax2d.set_ylim(-4.0 * res_scale, 1.5 * res_scale)
    ax2d.set_aspect('equal')
    ax2d.axis('off')
    plt.tight_layout()
    fig1.savefig(os.path.join(OUT, '_tmp_part1.png'), dpi=100 * res_scale, bbox_inches='tight')
    plt.close(fig1)

    # ── Part 2: 3D Spherical View (in-memory) ────────────────────────────
    fig2 = plt.figure(figsize=(8 * res_scale, 8 * res_scale))
    ax3d = fig2.add_subplot(111, projection='3d')
    fig2.patch.set_facecolor('white')
    ax3d.set_facecolor('white')
    ax3d.set_axis_off()
    ax3d.view_init(elev=25, azim=50)

    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax3d.plot_surface(x_sphere, y_sphere, z_sphere, color="k", alpha=0.03, zorder=0)

    r_cube = [-1, 1]
    for s, e in combinations(np.array(list(product(r_cube, r_cube, r_cube))), 2):
        if np.sum(np.abs(s - e)) == r_cube[1] - r_cube[0]:
            ax3d.plot3D(*zip(s, e), color="gray", alpha=0.4, linestyle=':')

    coords_unit = coords / np.linalg.norm(coords, axis=-1, keepdims=True)
    p1_unit = coords_unit[idx_front]
    p2_unit = coords_unit[idx_right]

    for face_idx in range(6):
        start = face_idx * points_per_face
        end = start + points_per_face
        face_points = coords_unit[start:end]
        X = face_points[:, 0].reshape(viz_grid, viz_grid)
        Y = face_points[:, 1].reshape(viz_grid, viz_grid)
        Z = face_points[:, 2].reshape(viz_grid, viz_grid)
        center = face_points.mean(axis=0)
        is_front = center.sum() > 0
        face_alpha = 0.5 if is_front else 0.1
        ax3d.plot_wireframe(X, Y, Z, color=colors[face_idx], alpha=face_alpha, linewidth=0.8 * res_scale)
        ax3d.plot_surface(X, Y, Z, color=colors[face_idx], alpha=0.15)

    draw_geodesic(ax3d, p1_unit, p2_unit, color='crimson', lw=2 * res_scale)
    ax3d.scatter(*p1_unit, color='red', s=60 * res_scale ** 2, edgecolors='black', zorder=100)
    ax3d.scatter(*p2_unit, color='darkred', s=60 * res_scale ** 2, edgecolors='black', zorder=100)
    mid_point = (p1_unit + p2_unit) / 2
    mid_point = mid_point / np.linalg.norm(mid_point) * 1.35
    ax3d.text(
        mid_point[0], mid_point[1], mid_point[2],
        r'$\mathcal{d}_g(p_i, p_j)$', color='crimson',
        fontweight='bold', fontsize=16 * res_scale,
    )
    ax3d.set_title(r"Restored 3D Spherical Topology", fontsize=18 * res_scale, fontweight='bold', y=0.95)
    limit = 1.35
    ax3d.set_xlim([-limit, limit])
    ax3d.set_ylim([-limit, limit])
    ax3d.set_zlim([-limit, limit])
    ax3d.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig2.savefig(os.path.join(OUT, '_tmp_part2.png'), dpi=100 * res_scale, bbox_inches='tight')
    plt.close(fig2)

    # ── Part 3: RoPE Correlation Heatmap (in-memory) ─────────────────────
    fig3 = plt.figure(figsize=(6 * res_scale, 6 * res_scale))
    ax_heat = fig3.add_subplot(111)
    fig3.patch.set_facecolor('white')
    ax_heat.set_title(r"Implicit 3D RoPE Correlation from $p_i$", fontsize=18 * res_scale, fontweight='bold')

    head_dim = model.hidden_size // model.transformer_encoder.transformer[0].attention.num_heads
    base_dim  = (head_dim // 6) * 2
    half_dim  = base_dim // 2

    if freq_scale is None:
        freq_scale = 4.0 * np.pi

    if ckpt_path is None:
        ckpt_path = os.path.join(REPO, 'cviq_fold3_best_checkpoint.pth')
    loaded_learned = False

    if os.path.exists(ckpt_path):
        try:
            _ckpt_full = torch.load(ckpt_path, map_location='cpu')
            state_dict = _ckpt_full.get('model_state_dict', _ckpt_full.get('ema_state_dict', _ckpt_full))

            num_heads = model.transformer_encoder.transformer[0].attention.num_heads
            head_dim_total = model.hidden_size // num_heads

            rope_corr = np.zeros(coords.shape[0])
            layer = 1
            W_q = state_dict[f'transformer_encoder.transformer.{layer}.attention.query.weight'].numpy()
            W_k = state_dict[f'transformer_encoder.transformer.{layer}.attention.key.weight'].numpy()

            inv_freq = 1.0 / (10000.0 ** (np.arange(0, half_dim) / half_dim))
            inv_freq *= freq_scale

            for head in range(num_heads):
                W_q_head = W_q[head * head_dim_total: (head + 1) * head_dim_total]
                W_k_head = W_k[head * head_dim_total: (head + 1) * head_dim_total]

                alpha = np.zeros((3, half_dim))
                beta = np.zeros((3, half_dim))
                for axis in range(3):
                    axis_offset = axis * 32
                    for m in range(half_dim):
                        d1 = axis_offset + 2 * m
                        d2 = axis_offset + 2 * m + 1
                        alpha[axis, m] = np.dot(W_q_head[d1], W_k_head[d1]) + np.dot(W_q_head[d2], W_k_head[d2])
                        beta[axis, m] = np.dot(W_q_head[d2], W_k_head[d1]) - np.dot(W_q_head[d1], W_k_head[d2])

                if use_window:
                    w = np.hanning(half_dim)
                    for axis in range(3):
                        alpha[axis] *= w
                        beta[axis]  *= w

                head_corr = np.zeros(coords.shape[0])
                for axis in range(3):
                    theta_i = p1[axis] * inv_freq
                    theta_j = coords[:, axis:axis+1] * inv_freq[np.newaxis, :]
                    diff = theta_j - theta_i[np.newaxis, :]
                    axis_corr = np.sum(alpha[axis][np.newaxis, :] * np.cos(diff) + beta[axis][np.newaxis, :] * np.sin(diff), axis=1)
                    head_corr += axis_corr

                head_corr = head_corr / (head_corr[idx_front] + 1e-8)
                head_corr = head_corr - head_corr.mean()
                rope_corr += head_corr

            rope_corr /= num_heads
            loaded_learned = True
        except Exception as e:
            print(f'  Checkpoint load failed: {e}, using fallback')

    if not loaded_learned:
        inv_freq = np.linspace(0.5, 4.0, half_dim) * freq_scale / (2.0 * np.pi)
        rope_corr = np.zeros(coords.shape[0])
        for axis in range(3):
            theta_i = p1[axis] * inv_freq
            theta_j = coords[:, axis:axis+1] * inv_freq[np.newaxis, :]
            diff = theta_j - theta_i[np.newaxis, :]
            rope_corr += np.mean(np.cos(diff), axis=1)
        rope_corr /= 3.0
        rope_corr = rope_corr / np.abs(rope_corr).max()

    corr_abs_max = max(abs(rope_corr.min()), abs(rope_corr.max()))
    vmin_global = -corr_abs_max
    vmax_global = corr_abs_max
    cmap = 'RdBu_r'

    spacing = 0.5
    for face_idx in range(6):
        face_grid = {
            0: (1, 2), 1: (1, 0), 2: (2, 1), 3: (0, 1), 4: (1, 1), 5: (1, 3),
        }
        row, col = face_grid[face_idx]
        start_x = col * (viz_grid + spacing)
        start_y = (2 - row) * (viz_grid + spacing)
        start = face_idx * points_per_face
        end = start + points_per_face
        face_corr = rope_corr[start:end].reshape(viz_grid, viz_grid)
        im = ax_heat.imshow(
            face_corr,
            extent=(start_x, start_x + viz_grid, start_y, start_y + viz_grid),
            origin='lower', cmap=cmap, alpha=0.9, vmin=vmin_global, vmax=vmax_global,
        )
        rect = Rectangle(
            (start_x, start_y), viz_grid, viz_grid,
            linewidth=1.5 * res_scale, edgecolor='black', facecolor='none', zorder=50,
        )
        ax_heat.add_patch(rect)

    ax_heat.set_xlim(-1, 4 * (viz_grid + spacing) + 0.5)
    ax_heat.set_ylim(-1, 3 * (viz_grid + spacing) + 0.5)
    ax_heat.set_aspect('equal')
    ax_heat.axis('off')

    ax_heat.scatter(x_f_2d, y_f_2d, color='red', s=80 * res_scale ** 2, edgecolors='white', zorder=100)
    ax_heat.scatter(x_r_2d, y_r_2d, color='darkred', s=80 * res_scale ** 2, edgecolors='white', zorder=100)
    ax_heat.text(x_f_2d, y_f_2d + 1.5, r'$p_i$',
                 color='white', fontsize=16 * res_scale, ha='center', fontweight='bold')
    ax_heat.text(x_r_2d, y_r_2d + 1.5, r'$p_j$',
                 color='white', fontsize=16 * res_scale, ha='center', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.ax.tick_params(labelsize=12 * res_scale)
    if loaded_learned:
        cbar.set_label(
            r'RoPE Positional Correlation (Layer 1, averaged over heads)',
            fontsize=14 * res_scale,
        )
        tick_val = float(f'{vmax_global:.2f}')
        cbar.set_ticks([-tick_val, -tick_val / 2, 0, tick_val / 2, tick_val])
    else:
        cbar.set_label(r'RoPE Positional Correlation (low-freq)', fontsize=14 * res_scale)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    plt.tight_layout()
    fig3.savefig(os.path.join(OUT, '_tmp_part3.png'), dpi=100 * res_scale, bbox_inches='tight')
    plt.close(fig3)

    # ── Join into combined layout ─────────────────────────────────────────
    image_files = [
        os.path.join(OUT, '_tmp_part1.png'),
        os.path.join(OUT, '_tmp_part2.png'),
        os.path.join(OUT, '_tmp_part3.png'),
    ]
    images = [crop_margins(f, margin=10) for f in image_files]

    top_img = images[0]
    bot_left_img = images[1]
    bot_right_img = images[2]

    # Match bottom image heights so titles align
    bottom_img_height = max(bot_left_img.height, bot_right_img.height)
    if bot_left_img.height != bottom_img_height:
        new_w = int(bot_left_img.width * (bottom_img_height / bot_left_img.height))
        bot_left_img = bot_left_img.resize((new_w, bottom_img_height), Image.Resampling.LANCZOS)
    if bot_right_img.height != bottom_img_height:
        new_w = int(bot_right_img.width * (bottom_img_height / bot_right_img.height))
        bot_right_img = bot_right_img.resize((new_w, bottom_img_height), Image.Resampling.LANCZOS)

    bottom_width = bot_left_img.width + bot_right_img.width
    bottom_height = bottom_img_height

    if top_img.width != bottom_width:
        new_height = int(top_img.height * (bottom_width / top_img.width))
        top_img = top_img.resize((bottom_width, new_height), Image.Resampling.LANCZOS)

    final_width = max(top_img.width, bottom_width)
    final_height = top_img.height + bottom_height

    joined = Image.new('RGB', (final_width, final_height), color='white')
    x_top = (final_width - top_img.width) // 2
    joined.paste(top_img, (x_top, 0))
    x_bottom_start = (final_width - bottom_width) // 2
    y_bottom_start = top_img.height
    joined.paste(bot_left_img, (x_bottom_start, y_bottom_start))
    joined.paste(bot_right_img, (x_bottom_start + bot_left_img.width, y_bottom_start))

    out_path = os.path.join(OUT, 'fig9_3d_mapping.jpg')
    joined.save(out_path, quality=95)

    # Clean up temp files
    for f in image_files:
        if os.path.exists(f):
            os.remove(f)

    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 10 * res_scale,
        'axes.titlesize': 12 * res_scale,
        'axes.labelsize': 11 * res_scale,
        'xtick.labelsize': 9 * res_scale,
        'ytick.labelsize': 9 * res_scale,
        'legend.fontsize': 9 * res_scale,
    })
    print(f'  Saved fig9_3d_mapping.jpg')


# ── Equirectangular projection diagram (imgs/3.jpg) ─────────────────────
def _render_sphere(W_out, H_out, tex, elev=25, azim=50, draw_grid=True,
                   sphere_rot=None):
    """Render a textured sphere with Plotly.

    sphere_rot: optional 3×3 rotation matrix applied to the sphere
                (keeps camera fixed, rotates the sphere in place).
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    import io

    tex_np = np.array(tex.convert('RGB'))
    tex_H, tex_W = tex_np.shape[:2]
    R = 1.0

    # Build dense mesh for texture mapping via vertex colors
    n_theta = 540
    n_phi = 270
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi, endpoint=True)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = R * np.sin(phi_grid) * np.cos(theta_grid)
    y = R * np.sin(phi_grid) * np.sin(theta_grid)
    z = R * np.cos(phi_grid)

    # Sample ERP texture at each vertex
    tx = ((theta_grid + np.pi) / (2 * np.pi) * (tex_W - 1)).astype(int) % tex_W
    ty = (phi_grid / np.pi * (tex_H - 1)).astype(int) % tex_H
    colors = tex_np[ty, tx] / 255.0

    x_f, y_f, z_f = x.flatten(), y.flatten(), z.flatten()
    c_f = colors.reshape(-1, 3).tolist()

    # Build triangle indices for the regular grid
    i_idx, j_idx, k_idx = [], [], []
    for row in range(n_phi - 1):
        for col in range(n_theta - 1):
            v00 = row * n_theta + col
            v01 = row * n_theta + col + 1
            v10 = (row + 1) * n_theta + col
            v11 = (row + 1) * n_theta + col + 1
            i_idx.extend([v00, v01])
            j_idx.extend([v01, v11])
            k_idx.extend([v10, v10])

    if sphere_rot is not None:
        coords = np.column_stack([x_f, y_f, z_f]) @ sphere_rot.T
        x_f, y_f, z_f = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x_f, y=y_f, z=z_f,
        i=i_idx, j=j_idx, k=k_idx,
        vertexcolor=c_f,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        lightposition=dict(x=0.0, y=0.0, z=1.0),
        showscale=False,
    ))

    if draw_grid:
        n_lon = 7
        n_lat = 7
        line_style = dict(color='white', width=max(3, W_out // 120))
        R_grid = R * 1.015
        # Longitude lines
        for i in range(n_lon):
            theta_i = i * 2 * np.pi / (n_lon - 1)
            phi_vals = np.linspace(0.01, np.pi - 0.01, 40)
            xg = R_grid * np.sin(phi_vals) * np.cos(theta_i)
            yg = R_grid * np.sin(phi_vals) * np.sin(theta_i)
            zg = R_grid * np.cos(phi_vals)
            if sphere_rot is not None:
                coords = np.column_stack([xg, yg, zg]) @ sphere_rot.T
                xg, yg, zg = coords[:, 0], coords[:, 1], coords[:, 2]
            fig.add_trace(go.Scatter3d(
                x=xg, y=yg, z=zg, mode='lines',
                line=line_style, showlegend=False, hoverinfo='none'
            ))
        # Latitude lines
        for i in range(n_lat):
            phi_i = i * np.pi / (n_lat - 1)
            theta_vals = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            xg = R_grid * np.sin(phi_i) * np.cos(theta_vals)
            yg = R_grid * np.sin(phi_i) * np.sin(theta_vals)
            zg = np.full_like(theta_vals, R_grid * np.cos(phi_i))
            if sphere_rot is not None:
                coords = np.column_stack([xg, yg, zg]) @ sphere_rot.T
                xg, yg, zg = coords[:, 0], coords[:, 1], coords[:, 2]
            fig.add_trace(go.Scatter3d(
                x=xg, y=yg, z=zg, mode='lines',
                line=line_style, showlegend=False, hoverinfo='none'
            ))

    # Camera setup (orthographic for tighter fill)
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    cam_dist = 1.8
    eye_x = cam_dist * np.cos(elev_rad) * np.sin(azim_rad)
    eye_y = cam_dist * np.cos(elev_rad) * np.cos(azim_rad)
    eye_z = cam_dist * np.sin(elev_rad)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.05, 1.05]),
            yaxis=dict(visible=False, range=[-1.05, 1.05]),
            zaxis=dict(visible=False, range=[-1.05, 1.05]),
            bgcolor='white',
            camera=dict(
                eye=dict(x=eye_x, y=eye_y, z=eye_z),
                projection=dict(type='orthographic'),
            ),
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        showlegend=False,
    )

    # Render to image, then crop to sphere and scale up to fill canvas
    buf = io.BytesIO()
    pio.write_image(fig, buf, format='png', width=W_out, height=H_out)
    img_bytes = buf.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    result = np.array(img.convert('RGB'))

    # Crop to non-white content (the sphere)
    bg = np.array([255, 255, 255])
    mask = (result != bg).any(axis=-1)
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = result[rmin:rmax + 1, cmin:cmax + 1]
        ch, cw = cropped.shape[:2]
        side = max(ch, cw)
        # Pad to square
        padded = np.full((side, side, 3), 255, dtype=np.uint8)
        y_off = (side - ch) // 2
        x_off = (side - cw) // 2
        padded[y_off:y_off + ch, x_off:x_off + cw] = cropped
        # Scale to fill 60% of the shorter dimension
        scale = min(W_out, H_out) / side * 0.7
        new_w = int(side * scale)
        new_h = int(side * scale)
        scaled = Image.fromarray(padded).resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Center on white canvas
        canvas = Image.new('RGB', (W_out, H_out), (255, 255, 255))
        canvas.paste(scaled, ((W_out - new_w) // 2, (H_out - new_h) // 2))
        return np.array(canvas)
    return result


def plot_equirectangular_diagram(res_scale=1.0):
    _render_scale = res_scale
    res_scale = res_scale * 0.5
    src_path = os.path.join(REPO, 'live/Graffiti_4k.png')
    pil_img = Image.open(src_path)
    W_full, H_full = pil_img.size
    pil_img = pil_img.crop((0, 0, W_full, H_full // 2))
    # Texture for the ERP panel display
    tex_w, tex_h = 1080, 540
    tex = pil_img.resize((tex_w, tex_h), Image.Resampling.LANCZOS)
    tex_np = np.array(tex)
    # Full-res texture for the sphere rendering
    tex_sph = pil_img  # 4096 × 1024

    fig = plt.figure(figsize=(10 * res_scale, 3.5 * res_scale))
    fig.patch.set_facecolor('white')

    # Right panel: flat equirectangular + grid + labels
    ax1 = fig.add_axes([0.38, 0.18, 0.58, 0.68])
    ax1.imshow(tex_np, extent=[0, tex_w, 0, tex_h])
    # Thick grid: longitude lines every 45°, latitude lines every 30°.
    # In a standard ERP, latitude lines are evenly spaced in the image,
    # which is the key visual — equal angular steps occupy equal pixel rows.
    n_lon = 7
    for i in range(n_lon):
        x = i * tex_w / (n_lon - 1)
        ax1.axvline(x, color='white', alpha=0.5, lw=1.5 * res_scale)
    n_lat = 7
    for i in range(n_lat):
        phi = -math.pi / 2 + i * math.pi / 6
        y = (-phi + math.pi / 2) / math.pi * tex_h
        ax1.axhline(y, color='white', alpha=0.5, lw=1.5 * res_scale)

    lon_labels = ['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E']
    for i, lbl in enumerate(lon_labels):
        x = i * tex_w / 6
        ax1.text(x, tex_h * 1.07, lbl, color='black', fontsize=6 * res_scale,
                 ha='center', va='top', fontfamily='serif',
                 clip_on=False)

    lat_labels = ['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S']
    for i, lbl in enumerate(lat_labels):
        phi = -(math.pi / 2) + i * math.pi / 6
        y = (-phi + math.pi / 2) / math.pi * tex_h
        ax1.text(tex_w * 1.03, y, lbl, color='black', fontsize=6 * res_scale,
                 va='center', ha='left', fontfamily='serif',
                 clip_on=False)

    ax1.set_xlim(-tex_w * 0.03, tex_w * 1.15)
    ax1.set_ylim(-tex_h * 0.05, tex_h * 1.11)
    ax1.set_title('Equirectangular Projection',
                  fontsize=11 * res_scale, fontweight='bold', fontfamily='serif', pad=6)
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax1.spines.values():
        s.set_visible(False)

    # Left panel: sphere rendered via direct texture projection
    graffiti = Image.open(os.path.join(REPO, 'live', 'Graffiti_4k.png')).convert('RGB')
    tex_w, tex_h = 1024, 512
    # Stereoscopic top-bottom: use top half only
    graffiti = graffiti.crop((0, 0, graffiti.width, graffiti.height // 2))
    tex_np = np.array(graffiti.resize((tex_w, tex_h), Image.LANCZOS))

    sph_out = 1024
    ys, xs = np.mgrid[0:sph_out, 0:sph_out]
    scale = sph_out / 2
    u = (xs - sph_out / 2) / scale
    v = -(ys - sph_out / 2) / scale

    # Camera: orthographic, looking at origin from (azim=50, elev=20)
    azim_r = np.radians(30)
    elev_r = np.radians(20)
    ca, sa = np.cos(azim_r), np.sin(azim_r)
    ce, se = np.cos(elev_r), np.sin(elev_r)
    cam_pos = np.array([ca * ce, sa * ce, se])
    view_dir = -cam_pos  # camera to scene

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(view_dir, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_dir)

    # For each pixel, ray: P = cam_pos + u*right + v*up + z*view_dir
    A0 = cam_pos + u[:, :, None] * right + v[:, :, None] * up
    A0_dot_F = np.sum(A0 * view_dir, axis=2)
    A0_dot_A = np.sum(A0 ** 2, axis=2)
    disc = A0_dot_F ** 2 - A0_dot_A + 1.0  # R_sph = 1
    valid = disc >= 0
    z = np.where(valid, -A0_dot_F - np.sqrt(np.maximum(disc, 0)), 0.0)
    P = A0 + z[:, :, None] * view_dir

    # Texture lookup
    theta = np.arctan2(P[:, :, 1], P[:, :, 0])
    phi = np.arccos(np.clip(P[:, :, 2], -1, 1))
    tx = ((theta / (2 * np.pi) + 0.5) * tex_w).astype(int) % tex_w
    ty = (phi / np.pi * tex_h).astype(int) % tex_h

    sph_rgb = np.full((sph_out, sph_out, 3), 255, dtype=np.uint8)
    sph_rgb[valid] = tex_np[ty[valid], tx[valid]]

# 3D UV grid — match ERP exactly: every 60° longitude, every 30° latitude
    grid_th = np.arange(6) * 2 * np.pi / 6   # 0, 60, 120, 180, 240, 300°
    grid_ph = np.arange(7) * np.pi / 6        # 0, 30, 60, 90, 120, 150, 180°
    tol = 2.0 / sph_out * 3  # 3-pixel tolerance in normalized UV space

    # theta and phi arrays from the texture projection above
    for gth in grid_th:
        dth = np.abs(np.arctan2(np.sin(theta - gth), np.cos(theta - gth)))
        mask = (dth < tol) & valid
        sph_rgb[mask] = [255, 255, 255]
    for gph in grid_ph:
        dph = np.abs(phi - gph)
        mask = (dph < tol) & valid
        sph_rgb[mask] = [255, 255, 255]

    # Pad with white border so sphere appears smaller within the axis
    pad = sph_out // 3
    padded = np.full((sph_out + 2*pad, sph_out + 2*pad, 3), 255, dtype=np.uint8)
    padded[pad:pad+sph_out, pad:pad+sph_out] = sph_rgb
    sph_rgb = padded

    # Sphere visual poles in padded image coords
    sph_north = (pad + sph_out // 2, pad)
    sph_south = (pad + sph_out // 2, pad + sph_out - 1)

    ax2 = fig.add_axes([0.10, 0.18, 0.28, 0.68])
    ax2.imshow(sph_rgb, interpolation='bilinear', aspect='equal')
    ax2.set_title('Sphere',
                  fontsize=11 * res_scale, fontweight='bold', fontfamily='serif', pad=6)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax2.spines.values():
        s.set_visible(False)

    # Arrows: sphere visual poles → ERP left corners
    # Use ConnectionPatch for per-axis data coords (render-time transforms)
    from matplotlib.patches import ConnectionPatch
    erp_h = 540  # extent was [0, 1080, 0, 540], tex_h got overwritten at line 1419
    for px, py, dst_data, col in [
        (sph_north[0], sph_north[1] - int(15 * _render_scale), (-20, erp_h), '#d62728'),
        (sph_south[0], sph_south[1] + int(15 * _render_scale), (-20, 0), '#2c7bb6')
    ]:
        fig.patches.append(ConnectionPatch(
            xyA=(px, py), coordsA=ax2.transData,
            xyB=dst_data, coordsB=ax1.transData,
            arrowstyle=f'->,head_width={0.4*res_scale:.1f},head_length={0.7*res_scale:.1f}',
            color=col, lw=1.2 * res_scale,
            zorder=100, clip_on=False,
            shrinkA=0, shrinkB=0
        ))

    fig.savefig(os.path.join(OUT, 'fig3_equirectangular.jpg'), dpi=250 * res_scale, pil_kwargs=dict(quality=95))
    plt.close(fig)
    # Crop white margins
    img = Image.open(os.path.join(OUT, 'fig3_equirectangular.jpg')).convert('RGB')
    arr = np.array(img)
    mask = np.any(arr < 230, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y = np.where(rows)[0]
    x = np.where(cols)[0]
    if len(y) and len(x):
        img = img.crop((x[0], y[0], x[-1] + 1, y[-1] + 1))
        img.save(os.path.join(OUT, 'fig3_equirectangular.jpg'), quality=95)
    print('  Saved fig3_equirectangular.jpg')


# ── Architecture Diagram ────────────────────────────────────────────────
def plot_architecture_diagram(res_scale=1.0):
    """Render the SpherIQ architecture diagram (arch.jpg) via Mermaid + puppeteer."""
    mmd = os.path.join(REPO, 'figures', 'spheriq_arch.mmd')
    tmp_svg = os.path.join(OUT, '_arch_tmp.svg')
    out_png = os.path.join(OUT, 'fig1_architecture.jpg')

    if not os.path.exists(mmd):
        print(f'  Skipping arch diagram: {mmd} not found')
        return

    puppeteer = '/usr/lib/node_modules/@mermaid-js/mermaid-cli/node_modules/puppeteer'
    chromium  = os.path.expanduser(
        '~/.cache/puppeteer/chrome-headless-shell/linux-150.0.7871.24/'
        'chrome-headless-shell-linux64/chrome-headless-shell'
    )

    # Write puppeteer config alongside the script (not in /tmp)
    pp_conf = os.path.join(os.path.dirname(__file__), '.puppeteer-config.json')
    with open(pp_conf, 'w') as f:
        json.dump({"args": ["--no-sandbox", "--disable-setuid-sandbox"]}, f)

    # Step 1: MMD → SVG via mmdc
    env = os.environ.copy()
    env['PUPPETEER_CACHE_DIR'] = '/home/luke/.cache/puppeteer'
    subprocess.run([
        'mmdc', '-i', mmd, '-o', tmp_svg,
        '-w', '1400', '-H', '1200',
        '--backgroundColor', 'white',
        '-p', pp_conf,
    ], check=True, capture_output=True, env=env)
    os.remove(pp_conf)

    # Step 2: Post-process SVG for border-radius
    with open(tmp_svg) as f:
        data = f.read()
    for m in re.finditer(r'<g[^>]*class="cluster"[^>]*>(.*?)</g>', data, re.DOTALL):
        inner = m.group(1)
        def _fix(t):
            tag = t.group(0)
            return tag[:-2] + ' rx="12" ry="12"/>' if tag.endswith('/>') else tag
        inner_fixed = re.sub(r'<rect[^>]*/>', _fix, inner)
        data = data.replace(inner, inner_fixed)
    def _node_rx(m):
        t = m.group(0)
        return t[:-2] + ' rx="4" ry="4"/>' if t.endswith('/>') else t
    data = re.sub(r'<rect[^>]*class="basic label-container"[^>]*/>', _node_rx, data)
    with open(tmp_svg, 'w') as f:
        f.write(data)

    # Step 3: Determine output size from SVG viewBox
    m = re.search(r'viewBox="[^"]*\s+\d+\s+([\d.]+)\s+([\d.]+)"', data)
    if m:
        svg_w, svg_h = float(m.group(1)), float(m.group(2))
    else:
        svg_w, svg_h = 1146, 1518
    out_w = int(max(1400, int(svg_w) + 200) * res_scale)
    out_h = int((svg_h * out_w / svg_w) + 100)

    # Step 4: SVG → PNG via headless Chrome
    js = f'''
    const puppeteer = require('{puppeteer}');
    const fs = require('fs');
    const svg = fs.readFileSync('{tmp_svg}', 'utf8');
    const b64 = Buffer.from(svg).toString('base64');
    const uri = 'data:image/svg+xml;base64,' + b64;
    (async () => {{
      const browser = await puppeteer.launch({{
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
        defaultViewport: {{ width: {out_w}, height: {out_h} }},
        executablePath: '{chromium}'
      }});
      const page = await browser.newPage();
      await page.setViewport({{ width: {out_w}, height: {out_h}, deviceScaleFactor: 1 }});
      const html = '<!DOCTYPE html><html><body style="margin:0;background:white">'
        + '<img src="' + uri + '" width="{out_w}" height="{out_h}" style="display:block">'
        + '</body></html>';
      await page.setContent(html, {{ waitUntil: 'networkidle0' }});
      await page.screenshot({{ path: '{out_png}', fullPage: true }});
      await browser.close();
    }})();
    '''
    subprocess.run([
        'node', '-e', js
    ], check=True, env={**os.environ,
        'NODE_PATH': '/usr/lib/node_modules/@mermaid-js/mermaid-cli/node_modules',
        'PUPPETEER_CACHE_DIR': os.path.expanduser('~/.cache/puppeteer'),
    })

    if os.path.exists(tmp_svg):
        os.remove(tmp_svg)
    print(f'  Saved fig1_architecture.jpg')

# ── Multi-Scale Patch Extraction Diagram (5.jpg) ────────────────────────
def plot_patch_extraction(fname='fig5_patches.jpg', src_basename='Graffiti_4k.png', res_scale=1.0):
    """Multi-scale patch extraction diagram.
    
    Args:
        fname: output filename (saved to OUT/)
        src_basename: source equirectangular image in REPO/live/
        res_scale: scale factor for all dimensions (default 1.0)
    """
    print(f'  Generating {fname} (src={src_basename}, res_scale={res_scale})...')
    res_scale = res_scale * 2  # internal boost for this figure
    src_path = os.path.join(REPO, 'live', src_basename)
    pil_img = Image.open(src_path)
    W_full, H_full = pil_img.size
    pil_img = pil_img.crop((0, 0, W_full, H_full // 2))
    base_tex_w, base_tex_h = 1080, 540
    tex_w = int(base_tex_w * res_scale)
    tex_h = int(base_tex_h * res_scale)
    tex_np = np.array(pil_img.resize((tex_w, tex_h), Image.Resampling.LANCZOS))
    Ht, Wt = tex_np.shape[:2]

    def sc(v):
        return max(1, int(v * res_scale))

    # Extract cubemap faces at nice tile size
    ts = sc(72)
    i, j = np.meshgrid(np.arange(ts), np.arange(ts))
    u = -1 + 2 * (i + 0.5) / ts
    v = -1 + 2 * (j + 0.5) / ts

    def _face(rx_fn, ry_fn, rz_fn):
        rx, ry, rz = rx_fn(), ry_fn(), rz_fn()
        norm = np.sqrt(rx*rx + ry*ry + rz*rz)
        rx, ry, rz = rx/norm, ry/norm, rz/norm
        theta = np.arctan2(rx, rz)
        phi = np.arccos(np.clip(ry, -1.0, 1.0))
        tx = np.floor((theta + np.pi) / (2 * np.pi) * Wt).astype(np.int32) % Wt
        ty = np.floor(phi / np.pi * Ht).astype(np.int32) % Ht
        out = tex_np[ty, tx, :]
        return np.transpose(out, (1, 0, 2))

    face_order = ['left', 'front', 'right', 'back', 'top', 'bottom']
    faces = {
        'front':  _face(lambda: v,   lambda: -u, lambda: np.ones_like(u)),
        'back':   _face(lambda: -v,  lambda: -u, lambda: -np.ones_like(u)),
        'right':  _face(lambda: np.ones_like(u),  lambda: -u, lambda: -v),
        'left':   _face(lambda: -np.ones_like(u), lambda: -u, lambda: v),
        'top':    _face(lambda: v,   lambda: np.ones_like(u),  lambda: u),
        'bottom': _face(lambda: v,   lambda: -np.ones_like(u), lambda: -u),
    }

    # Second set: yaw the ERP, extract faces from rotated view
    yaw_deg = 20
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    xs = np.arange(tex_w)
    ys = np.arange(tex_h)
    xx, yy = np.meshgrid(xs, ys)
    theta = xx.astype(np.float64) / tex_w * 2 * np.pi - np.pi
    phi = yy.astype(np.float64) / tex_h * np.pi
    sin_phi = np.sin(phi)
    vecs = np.stack([
        sin_phi * np.cos(theta),
        sin_phi * np.sin(theta),
        np.cos(phi)
    ], axis=-1)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vecs_r = vecs @ R.T
    theta_r = np.arctan2(vecs_r[:, :, 1], vecs_r[:, :, 0])
    phi_r = np.arccos(np.clip(vecs_r[:, :, 2], -1.0, 1.0))
    tx2 = np.floor((theta_r + np.pi) / (2 * np.pi) * tex_w).astype(np.int32) % tex_w
    ty2 = np.floor(phi_r / np.pi * tex_h).astype(np.int32) % tex_h
    tex2_np = tex_np[ty2, tx2, :]

    H2, W2 = tex2_np.shape[:2]
    faces2 = {}
    for name in face_order:
        fn = {
            'front':  lambda: (v,   -u, np.ones_like(u)),
            'back':   lambda: (-v,  -u, -np.ones_like(u)),
            'right':  lambda: (np.ones_like(u),  -u, -v),
            'left':   lambda: (-np.ones_like(u), -u, v),
            'top':    lambda: (v,   np.ones_like(u),  u),
            'bottom': lambda: (v,   -np.ones_like(u), -u),
        }[name]()
        rx, ry, rz = fn
        norm = np.sqrt(rx*rx + ry*ry + rz*rz)
        rx, ry, rz = rx/norm, ry/norm, rz/norm
        theta = np.arctan2(rx, rz)
        phi = np.arccos(np.clip(ry, -1.0, 1.0))
        tx = np.floor((theta + np.pi) / (2 * np.pi) * W2).astype(np.int32) % W2
        ty = np.floor(phi / np.pi * H2).astype(np.int32) % H2
        out = tex2_np[ty, tx, :]
        faces2[name] = np.transpose(out, (1, 0, 2))

    # Cross layout: 5 cols × 3 rows
    def tile_pil(fd):
        return Image.fromarray(fd.astype('uint8'), 'RGB')

    def make_cross_pil(faces_dict, positions, alpha=1.0, with_border=False):
        """Draw a single cross with flush tiles (no inter-tile gap)."""
        min_col = min(c for _, (c, _) in positions.items())
        max_col = max(c for _, (c, _) in positions.items())
        min_row = min(r for _, (_, r) in positions.items())
        max_row = max(r for _, (_, r) in positions.items())
        w = (max_col - min_col + 1) * ts
        h = (max_row - min_row + 1) * ts
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        dwg = ImageDraw.Draw(img) if with_border else None
        for name, (col, row) in positions.items():
            px = (col - min_col) * ts
            py = (row - min_row) * ts
            tile = tile_pil(faces_dict[name])
            if alpha < 1.0:
                tile = Image.blend(Image.new('RGB', tile.size, (255, 255, 255)), tile, alpha)
            img.paste(tile, (px, py))
            if dwg:
                dwg.rectangle([px, py, px + ts - 1, py + ts - 1], outline=with_border, width=sc(2))
        return img

    def make_grid_pil(tiles, cols, gap_px, tile_size=None, border_color='#555555'):
        """Arrange PIL Images into a grid with borders, return RGBA PIL Image."""
        ts_use = tile_size or ts
        rows = (len(tiles) + cols - 1) // cols
        gw = cols * ts_use + (cols - 1) * gap_px
        gh = rows * ts_use + (rows - 1) * gap_px
        img = Image.new('RGBA', (gw, gh), (0, 0, 0, 0))
        dwg = ImageDraw.Draw(img)
        for k, tile in enumerate(tiles):
            col = k % cols
            row = k // cols
            px = col * (ts_use + gap_px)
            py = row * (ts_use + gap_px)
            img.paste(tile, (px, py))
            dwg.rectangle([px, py, px + ts_use - 1, py + ts_use - 1], outline=border_color, width=sc(2))
        return img

    # ---- Composite the figure ----
    margin = sc(14)
    arr_gap = sc(22)
    arrow_w = sc(30)

    # Stage 1: two crosses on a shared canvas, flush tiles + gap between crosses
    cross1_pos = {
        'top':    (1, 0), 'bottom': (1, 2),
        'left':   (0, 1), 'front':  (1, 1),
        'right':  (2, 1), 'back':   (3, 1),
    }
    cross2_pos = {
        'top':    (2, 2), 'bottom': (2, 4),
        'left':   (1, 3), 'front':  (2, 3),
        'right':  (3, 3), 'back':   (4, 3),
    }
    cross1 = make_cross_pil(faces, cross1_pos)
    cross2 = make_cross_pil(faces, cross2_pos)
    cross_gap = sc(8)
    # cross2 is offset by (1 col, 2 rows) from cross1 + uniform gap
    c2x = ts + cross_gap
    c2y = 2 * ts + cross_gap
    s1_w = max(cross1.width, c2x + cross2.width)
    s1_h = max(cross1.height, c2y + cross2.height)
    s1_pil = Image.new('RGBA', (s1_w, s1_h), (0, 0, 0, 0))
    s1_pil.paste(cross1, (0, 0), cross1)
    s1_pil.paste(cross2, (c2x, c2y), cross2)

    # Stage 2: 3×4 grid sized to match stage 1 height
    all_12_pil = [tile_pil(faces[n]) for n in face_order] * 2
    s2_gap = sc(4)
    s2_rows = 4
    s2_target_h = max(cross1.height, c2y + cross2.height)
    s2_ts = (s2_target_h - (s2_rows - 1) * s2_gap) // s2_rows
    s2_tiles = [t.resize((s2_ts, s2_ts), Image.LANCZOS) for t in all_12_pil]
    s2_pil = make_grid_pil(s2_tiles, 3, s2_gap, tile_size=s2_ts)

    # Stage 3: multi-scale patches — constant size (32×32), fewer at coarser scales
    ps = sc(32)
    pg = sc(3)          # inter-patch gap within each grid
    gg = sc(10)         # gap between grids
    front_pil = tile_pil(faces['front'])

    # Scale 0 (blue): 4×4 = 16 patches (finest = most), top row, centered
    # Scale 1 (red): 3×3 = 9 patches, bottom-left
    # Scale 2 (orange): 2×2 = 4 patches (coarsest = fewest), bottom-right
    s0_rc, s1_rc, s2_rc = 4, 3, 2
    s0_gw = s0_rc * ps + (s0_rc - 1) * pg
    s0_gh = s0_rc * ps + (s0_rc - 1) * pg
    s1_gw = s1_rc * ps + (s1_rc - 1) * pg
    s1_gh = s1_rc * ps + (s1_rc - 1) * pg
    s2_gw = s2_rc * ps + (s2_rc - 1) * pg
    s2_gh = s2_rc * ps + (s2_rc - 1) * pg

    bot_w = s1_gw + gg + s2_gw
    s3_w = max(s0_gw, bot_w)
    s3_h = s0_gh + gg + max(s1_gh, s2_gh)
    s3_pil = Image.new('RGBA', (s3_w, s3_h), (0, 0, 0, 0))

    # Center scale 0 above the bottom row
    s0_ox = (s3_w - s0_gw) // 2

    def paste_scale(dst, rows_cols, ox, oy, color, alpha=1.0, seed=42):
        rr = np.random.RandomState(seed)
        for k in range(rows_cols * rows_cols):
            col, row = k % rows_cols, k // rows_cols
            x = ox + col * (ps + pg)
            y = oy + row * (ps + pg)
            sy = rr.randint(0, ts - ps + 1)
            sx = rr.randint(0, ts - ps + 1)
            patch = front_pil.crop((sx, sy, sx + ps, sy + ps))
            if alpha < 1.0:
                patch = Image.blend(Image.new('RGB', patch.size, (255, 255, 255)), patch, alpha)
            dst.paste(patch, (x, y))
            draw = ImageDraw.Draw(dst)
            draw.rectangle([x, y, x + ps - 1, y + ps - 1], outline=color, width=sc(2))

    paste_scale(s3_pil, s0_rc, s0_ox, 0, '#2c7bb6', seed=42)
    paste_scale(s3_pil, s1_rc, 0, s0_gh + gg, '#d7191c', alpha=0.85, seed=43)
    paste_scale(s3_pil, s2_rc, s1_gw + gg, s0_gh + gg, '#fdae61', alpha=0.75, seed=44)

    # ---- Full figure layout ----
    # [stage1 unified grid] [arrow] [stage2 grid] [arrow] [stage3 clusters]
    s1_w = s1_pil.width
    s1_h = s1_pil.height
    s2_w = s2_pil.width
    s2_h = s2_pil.height
    s3_w = s3_pil.width
    s3_h = s3_pil.height

    total_w = margin + s1_w + arr_gap + arrow_w + arr_gap + s2_w + arr_gap + arrow_w + arr_gap + s3_w + margin

    max_h = max(s1_h, s2_h, s3_h)
    total_h = max_h + margin * 2

    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)

    # Paste stages centered vertically
    s1_x = margin
    s1_y = margin + (max_h - s1_h) // 2
    canvas.paste(s1_pil, (s1_x, s1_y), s1_pil)

    # Arrow 1
    arr1_x = s1_x + s1_w + arr_gap
    arr_y = margin + max_h // 2
    ah = sc(16)
    aw = arrow_w - sc(4)
    ax_pts = [
        (arr1_x + aw, arr_y),
        (arr1_x, arr_y - ah // 2),
        (arr1_x, arr_y + ah // 2),
    ]
    draw.polygon(ax_pts, fill='#aaaaaa')

    # Stage 2
    s2_x = arr1_x + arrow_w + arr_gap
    s2_y = margin + (max_h - s2_h) // 2
    canvas.paste(s2_pil, (s2_x, s2_y), s2_pil)

    # Arrow 2
    arr2_x = s2_x + s2_w + arr_gap
    ax_pts2 = [
        (arr2_x + aw, arr_y),
        (arr2_x, arr_y - ah // 2),
        (arr2_x, arr_y + ah // 2),
    ]
    draw.polygon(ax_pts2, fill='#aaaaaa')

    # Stage 3
    s3_x = arr2_x + arrow_w + arr_gap
    s3_y = margin + (max_h - s3_h) // 2
    canvas.paste(s3_pil, (s3_x, s3_y), s3_pil)

    canvas = canvas.resize((canvas.width // 2, canvas.height // 2), Image.LANCZOS)
    canvas.save(f'{OUT}/{fname}', quality=95)
    print(f'  Saved {fname}')

# ── Main ────────────────────────────────────────────────────────────────
def main(res_scale=2.0):
    _scale_rc(res_scale)

    cv_path = os.path.join(PRED, 'cross_validate_all_results.csv')
    if not os.path.exists(cv_path):
        print(f'Error: {cv_path} not found')
        return
    df = pd.read_csv(cv_path)

    print('Generating all figures...\n')

    # Only generate figures referenced by LaTeX
    plot_loss_curves(res_scale=res_scale)
    plot_live_scatter(res_scale=res_scale)
    plot_ablation(res_scale=res_scale)
    plot_geom_ablation(res_scale=res_scale)
    plot_cross_dataset_bars(df, res_scale=res_scale)

    plot_dual_heatmap(df, 'fig12_heatmap.jpg', res_scale=res_scale)
    plot_zero_shot_scatter(res_scale=res_scale)
    plot_cviq_ablation(res_scale=res_scale)

    plot_3d_mapping(res_scale=res_scale)
    plot_equirectangular_diagram(res_scale=res_scale)
    plot_spherical_rotations(res_scale=res_scale)
    plot_cubemap_conversion(res_scale=res_scale)

    try:
        plot_architecture_diagram(res_scale=res_scale)
    except Exception:
        print('  Warning: architecture diagram generation failed (known mmdc issue)')
    plot_spherical_rotation_diagram(res_scale=res_scale)
    plot_patch_extraction(res_scale=res_scale)

    # Crop white margins on all final figure images (fixed pixel margin, not scaled)
    OUT = os.path.join(os.path.dirname(__file__), 'spheriq_0/imgs')
    for f in os.listdir(OUT):
        if f.endswith('.jpg') or f.endswith('.png'):
            crop_margins(os.path.join(OUT, f))

    # Clean up any stale files not referenced by LaTeX
    import glob
    OUT = os.path.join(os.path.dirname(__file__), 'spheriq_0/imgs')
    kept = {'fig1_architecture','fig2_rotations','fig3_equirectangular','fig4_cubemap',
            'fig5_patches','fig7_training_loss','fig9_3d_mapping',
            'fig10_ablation','fig11_geom_ablation','fig12_heatmap','fig13_validation',
            'fig14_transfer','fig_sphere',
            'live_validation_correlation','comparison','geom_ablation','pre_cross'}
    for f in os.listdir(OUT):
        name = os.path.splitext(f)[0]
        if name not in kept:
            os.remove(os.path.join(OUT, f))
            print(f'  Cleaned up: {f}')

    # Crop white margins on all final figure images (fixed pixel margin, not scaled)
    for f in os.listdir(OUT):
        if f.endswith('.jpg') or f.endswith('.png'):
            crop_margins(os.path.join(OUT, f))
            print(f'  Cropped margins: {f}')


def plot_cubemap_conversion(res_scale=1.0):
    """4.jpg — Equirectangular to Cubemap conversion diagram."""
    print('  Generating fig4_cubemap (cubemap conversion diagram)...')
    src_path = os.path.join(REPO, 'live/Graffiti_4k.png')
    pil_img = Image.open(src_path)
    W_full, H_full = pil_img.size
    pil_img = pil_img.crop((0, 0, W_full, H_full // 2))
    tex_w, tex_h = pil_img.size
    tex_np = np.array(pil_img)

    # Extract cubemap faces (vectorized)
    face_size = int(200 * res_scale)
    i, j = np.meshgrid(np.arange(face_size), np.arange(face_size))
    u = -1 + 2 * (i + 0.5) / face_size
    v = -1 + 2 * (j + 0.5) / face_size

    H, W = tex_np.shape[:2]

    def _face(rx_fn, ry_fn, rz_fn):
        rx, ry, rz = rx_fn(), ry_fn(), rz_fn()
        norm = np.sqrt(rx*rx + ry*ry + rz*rz)
        rx, ry, rz = rx/norm, ry/norm, rz/norm
        theta = np.arctan2(rx, rz)
        phi = np.arccos(np.clip(ry, -1.0, 1.0))
        tx = np.floor((theta + np.pi) / (2 * np.pi) * W).astype(np.int32) % W
        ty = np.floor(phi / np.pi * H).astype(np.int32) % H
        out = tex_np[ty, tx, :]
        return np.transpose(out, (1, 0, 2))

    faces = {
        'front':  _face(lambda: v,   lambda: -u, lambda: np.ones_like(u)),
        'back':   _face(lambda: -v,  lambda: -u, lambda: -np.ones_like(u)),
        'right':  _face(lambda: np.ones_like(u),  lambda: -u, lambda: -v),
        'left':   _face(lambda: -np.ones_like(u), lambda: -u, lambda: v),
        'top':    _face(lambda: v,   lambda: np.ones_like(u),  lambda: u),
        'bottom': _face(lambda: v,   lambda: -np.ones_like(u), lambda: -u),
    }

    colors = {
        'front': '#E74C3C', 'back': '#3498DB',
        'right': '#2ECC71', 'left': '#F39C12',
        'top': '#9B59B6', 'bottom': '#1ABC9C',
    }

    # Build face boundary polygons on the ERP (lines + translucent fill)
    inset = 0.03
    eqs = {
        'front':  lambda u, v: (v, -u, 1),
        'back':   lambda u, v: (-v, -u, -1),
        'right':  lambda u, v: (1, -u, -v),
        'left':   lambda u, v: (-1, -u, v),
        'top':    lambda u, v: (v, 1, u),
        'bottom': lambda u, v: (v, -1, -u),
    }

    from matplotlib.patches import Polygon

    def _face_data(face_name):
        """Return (tx_raw, ty_raw) and (tx_plot, ty_plot) for a face.
        Raw is unwrapped for fill; plot has NaNs at seams."""
        fn = eqs[face_name]
        n = 200
        lo, hi = -1 + inset, 1 - inset
        edge_defs = [
            (np.linspace(lo, hi, n), np.full(n, lo)),
            (np.full(n, hi), np.linspace(lo, hi, n)),
            (np.linspace(hi, lo, n), np.full(n, hi)),
            (np.full(n, lo), np.linspace(hi, lo, n)),
        ]
        all_tx, all_ty = [], []
        for eu, ev in edge_defs:
            rx, ry, rz = fn(eu, ev)
            norm = np.sqrt(rx*rx + ry*ry + rz*rz)
            rx, ry, rz = rx/norm, ry/norm, rz/norm
            theta = np.arctan2(rx, rz)
            phi = np.arccos(np.clip(ry, -1.0, 1.0))
            tx = ((theta + np.pi) / (2 * np.pi) * tex_w)
            ty = (phi / np.pi * tex_h)
            all_tx.append(tx)
            all_ty.append(ty)
        tx_raw = np.concatenate(all_tx)
        ty_raw = np.concatenate(all_ty)
        # Unwrap for fill
        tx_fill = tx_raw.copy()
        d = np.diff(tx_fill)
        for i in range(len(d)):
            if d[i] > tex_w / 2:
                tx_fill[i+1:] -= tex_w
            elif d[i] < -tex_w / 2:
                tx_fill[i+1:] += tex_w
        ty_fill = ty_raw.copy()
        # Top/bottom faces are centered on a pole, so their traced boundary
        # winds a FULL 360 degrees in theta (all longitudes meet at the
        # pole), unlike the side faces whose boundary spans a limited angular
        # range.  Closing the curve's end straight back to its start collapses
        # the fill into a thin sliver.  Fix: extend the polygon out to the
        # image's top/bottom edge before closing it.
        if face_name in ('top', 'bottom'):
            cap_y = 0.0 if face_name == 'top' else tex_h
            tx_fill = np.concatenate([tx_fill, [tx_fill[-1], tx_fill[0]]])
            ty_fill = np.concatenate([ty_fill, [cap_y, cap_y]])
        # Wrapped + NaNs for plot
        tx_plot = tx_raw % tex_w
        ty_plot = ty_raw.copy()
        breaks = np.where(np.abs(np.diff(tx_plot)) > tex_w / 2)[0] + 1
        for b in reversed(breaks):
            tx_plot = np.insert(tx_plot, b, np.nan)
            ty_plot = np.insert(ty_plot, b, np.nan)
        return (tx_fill, ty_fill), (tx_plot, ty_plot)

    # ------------------------------------------------------------------
    # LAYOUT — vertical stack: ERP on top, cubemap cross below
    #
    # Everything below is computed in *inches* rather than axes-fraction.
    # Working in inches and sizing the ERP box to exactly match the image
    # aspect ratio avoids letterboxing issues with imshow.
    # ------------------------------------------------------------------
    margin = 0.18 * res_scale
    title_h = 0.34 * res_scale
    gap = 0.25 * res_scale

    # Figure width matches ERP at a chosen height, cross fits within it.
    target_erp_h = 2.6 * res_scale
    erp_w = target_erp_h * (tex_w / tex_h)
    fig_w = erp_w + 2 * margin

    # Cross: 4 columns × 3 rows (horizontal cross), gap matches border width
    cross_cols, cross_rows = 4, 3
    grid_gap = (2.5 * res_scale) / 72  # border thickness in inches
    face_box = (erp_w - (cross_cols - 1) * grid_gap) / cross_cols
    cross_h = cross_rows * face_box + (cross_rows - 1) * grid_gap

    fig_h = 2 * margin + title_h + target_erp_h + gap + cross_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    def fx(v):
        return v / fig_w

    def fy(v):
        return v / fig_h

    # ERP — centered horizontally, at the top
    erp_left = (fig_w - erp_w) / 2
    erp_bottom = fig_h - margin - target_erp_h - title_h
    ax_erp = fig.add_axes([fx(erp_left), fy(erp_bottom), fx(erp_w), fy(target_erp_h)])
    ax_erp.imshow(tex_np)
    ax_erp.set_xlim(0, tex_w)
    ax_erp.set_ylim(tex_h, 0)
    ax_erp.set_autoscale_on(False)
    for name in faces:
        (tx_f, ty_f), (tx_p, ty_p) = _face_data(name)
        c = colors[name]
        poly = Polygon(np.column_stack([tx_f, ty_f]),
                       facecolor=c, alpha=0.35, lw=0,
                       clip_on=True)
        ax_erp.add_patch(poly)
        for shift in (-tex_w, tex_w):
            poly2 = Polygon(np.column_stack([tx_f + shift, ty_f]),
                           facecolor=c, alpha=0.35, lw=0,
                           clip_on=True)
            ax_erp.add_patch(poly2)
        ax_erp.plot(tx_p, ty_p, color=c, lw=2.2 * res_scale)
    ax_erp.axis('off')
    fig.text(
        fig_w / 2 / fig_w,
        (erp_bottom + target_erp_h + title_h * 0.7) / fig_h,
        'Equirectangular', fontsize=11 * res_scale,
        ha='center', va='center', fontweight='bold',
    )

    # Cubemap cross — centered below the ERP
    cross_left = (fig_w - (cross_cols * face_box + (cross_cols - 1) * grid_gap)) / 2
    cross_bottom = margin

    cross_w = cross_cols * face_box + (cross_cols - 1) * grid_gap
    fig.text(
        (cross_left + cross_w / 2 + face_box + grid_gap) / fig_w,
        (cross_bottom + cross_h + title_h * 0.2 - (face_box + grid_gap) / 2) / fig_h,
        'Cubemap Faces', fontsize=11 * res_scale,
        ha='center', va='center', fontweight='bold',
    )

    # Horizontal cross: top row = top, middle row = left/front/right/back, bottom row = bottom
    positions = {
        'top':    (1, 2),
        'left':   (0, 1), 'front': (1, 1), 'right': (2, 1), 'back': (3, 1),
        'bottom': (1, 0),
    }
    for name, (col, row) in positions.items():
        x0 = cross_left + col * (face_box + grid_gap)
        y0 = cross_bottom + row * (face_box + grid_gap)
        ax = fig.add_axes([fx(x0), fy(y0), fx(face_box), fy(face_box)])
        ax.imshow(faces[name])
        for spine in ax.spines.values():
            spine.set_color(colors[name])
            spine.set_linewidth(2.5 * res_scale)
        ax.set_xticks([])
        ax.set_yticks([])


    fig.savefig(f'{OUT}/fig4_cubemap.jpg', dpi=75 * res_scale, facecolor='white')
    plt.close(fig)
    print('  Saved fig4_cubemap.jpg')

def plot_spherical_rotations(res_scale=1.0):
    """2.jpg — 3-panel montage: ERP with spherical yaw rotation (horizontal shift)."""
    print('  Generating fig2_rotations (spherical rotation augmentation samples)...')
    src_path = os.path.join(REPO, 'live/Graffiti_4k.png')
    pil_img = Image.open(src_path)
    W_full, H_full = pil_img.size
    # Top half only (equator to north pole — interesting content)
    pil_img = pil_img.crop((0, 0, W_full, H_full // 2))
    tex = np.array(pil_img)

    def _apply_yaw(tex_np, yaw_deg):
        """Rotate the sphere around the vertical axis (z), return the shifted ERP."""
        H, W = tex_np.shape[:2]
        yaw = math.radians(yaw_deg)
        xs = np.arange(W)
        ys = np.arange(H)
        xx, yy = np.meshgrid(xs, ys)
        theta = xx.astype(np.float64) / W * 2 * np.pi - np.pi
        phi = yy.astype(np.float64) / H * np.pi
        sin_phi = np.sin(phi)
        vecs = np.stack([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            np.cos(phi)
        ], axis=-1)
        # Z-axis rotation (yaw): shift longitude
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        vecs_r = vecs @ R.T
        theta_r = np.arctan2(vecs_r[:, :, 1], vecs_r[:, :, 0])
        phi_r = np.arccos(np.clip(vecs_r[:, :, 2], -1.0, 1.0))
        tx = np.floor((theta_r + np.pi) / (2 * np.pi) * W).astype(np.int32) % W
        ty = np.floor(phi_r / np.pi * H).astype(np.int32) % H
        return tex_np[ty, tx, :]

    rotations = []
    for deg in [0, 120, 240]:
        rotations.append(_apply_yaw(tex, deg))

    labels = ['Original (Yaw=0°)', 'Yaw=120°', 'Yaw=240°']
    fig, axes = plt.subplots(1, 3, figsize=(12 * res_scale, 3.5 * res_scale), facecolor='white')
    for ax, arr, label in zip(axes, rotations, labels):
        ax.imshow(arr)
        ax.set_title(label, fontsize=12 * res_scale)
        ax.axis('off')
    plt.subplots_adjust(wspace=0.03, left=0.005, right=0.995, bottom=0.005, top=0.9)
    fig.savefig(f'{OUT}/fig2_rotations.jpg', dpi=75 * res_scale, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)
    print(f'  Saved fig2_rotations.jpg')


# ── Spherical Rotation Diagram (sphere.jpg) ─────────────────────────────
def plot_spherical_rotation_diagram(res_scale=1.0, sphere_r=1.0):
    """sphere.jpg — Wireframe sphere with rotation axis for data augmentation."""
    print('  Generating fig_sphere (spherical rotation diagram)...')
    R = sphere_r

    fig = plt.figure(figsize=(5 * res_scale, 5 * res_scale))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)

    # ── Sphere surface (semi-transparent white) ──────────────────────────
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 18)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    # ── Wireframe grid lines ────────────────────────────────────────────
    v_lines = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    u_fine = np.linspace(0, 2 * np.pi, 60)
    gr = 1.00 * R
    for phi in v_lines:
        ax.plot(gr * np.cos(u_fine) * np.sin(phi),
                gr * np.sin(u_fine) * np.sin(phi),
                gr * np.ones_like(u_fine) * np.cos(phi),
                color='#d0d0d0', linewidth=0.8 * res_scale, zorder=2)
    # Silhouette ring (sphere outline from current camera angle)
    elev_rad = np.radians(15)
    azim_rad = np.radians(45)
    V = np.array([np.cos(elev_rad) * np.cos(azim_rad),
                  np.cos(elev_rad) * np.sin(azim_rad),
                  np.sin(elev_rad)])
    if abs(V[0]) < 0.9:
        A = np.cross(V, [1, 0, 0])
    else:
        A = np.cross(V, [0, 1, 0])
    A /= np.linalg.norm(A)
    B = np.cross(V, A)
    t_circ = np.linspace(0, 2 * np.pi, 100)
    sil_x = R * (np.cos(t_circ) * A[0] + np.sin(t_circ) * B[0])
    sil_y = R * (np.cos(t_circ) * A[1] + np.sin(t_circ) * B[1])
    sil_z = R * (np.cos(t_circ) * A[2] + np.sin(t_circ) * B[2])
    ax.plot(sil_x, sil_y, sil_z, color='#d0d0d0', linewidth=0.8 * res_scale, zorder=0)

    # ── Three axes ──────────────────────────────────────────────────────
    ext_xy = 2.0 * R
    ext_z = 1.5 * R
    label_off = 0.15 * R
    lw_ax = 2.0 * res_scale
    fs_lab = 12 * res_scale

    axes_data = [
        (-ext_xy, 0, 0, ext_xy, 0, 0, '#2c7bb6', 'X'),
        (0, -ext_xy, 0, 0, ext_xy, 0, '#41ab5d', 'Y'),
        (0, 0, -ext_z, 0, 0, ext_z, '#d62728', 'Z'),
    ]
    for x1, y1, z1, x2, y2, z2, col, lab in axes_data:
        ax.plot([x1, x2], [y1, y2], [z1, z2], color=col, linewidth=lw_ax, alpha=0.5, zorder=5)
    ax.text(ext_xy + label_off, label_off, label_off, 'X', color='#2c7bb6',
            fontsize=fs_lab, fontweight='bold', ha='center', va='center', zorder=10)
    ax.text(label_off, ext_xy + label_off, label_off, 'Y', color='#41ab5d',
            fontsize=fs_lab, fontweight='bold', ha='center', va='center', zorder=10)
    ax.text(0, 0, ext_z - label_off, 'Z', color='#d62728',
            fontsize=fs_lab, fontweight='bold', ha='center', va='center', zorder=10)

    # ── Layout ──────────────────────────────────────────────────────────
    lim = 2.3 * R

    # ── Rotation arcs with 2D arrowheads (on top of everything) ─────────
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch

    class _Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs
        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return min(zs3d)

    def rotation_arc(axis, t0, t1, n=50, rr=1.2, color='#d62728', flip=False):
        ts = np.linspace(t0, t1, n)
        ar = rr * R
        if axis == 'z':
            xa, ya, za = ar * np.cos(ts), ar * np.sin(ts), np.zeros_like(ts)
        elif axis == 'y':
            xa, ya, za = ar * np.cos(ts), np.zeros_like(ts), ar * np.sin(ts)
        else:
            xa, ya, za = np.zeros_like(ts), ar * np.cos(ts), ar * np.sin(ts)
        ax.plot(xa[:-4], ya[:-4], za[:-4], color=color, linewidth=2.0 * res_scale, zorder=20)
        tail = np.array([xa[-5], ya[-5], za[-5]])
        tip = np.array([xa[-1], ya[-1], za[-1]])
        ax.add_artist(_Arrow3D(
            [tail[0], tip[0]], [tail[1], tip[1]], [tail[2], tip[2]],
            arrowstyle='-|>', color=color, lw=2.0 * res_scale, mutation_scale=15 * res_scale,
            shrinkA=0, shrinkB=0, zorder=25,
        ))

    rotation_arc('z', np.pi / 12, 2 * np.pi / 3, color='#d62728')
    rotation_arc('x', np.pi / 6, 5 * np.pi / 6, color='#2c7bb6')
    rotation_arc('y', -np.pi / 4, np.pi / 4, color='#41ab5d')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f'{OUT}/fig_sphere.jpg', dpi=150 * res_scale, facecolor='white')
    plt.close(fig)
    print('  Saved fig_sphere.jpg')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-scale', type=float, default=2.0, help='Resolution scale factor (default: 2.0)')
    args = parser.parse_args()
    main(res_scale=args.res_scale)
