#!/usr/bin/env python3
"""
Generate publication-quality figures for the MUSIQ-VR cross-dataset paper.
Outputs PDFs suitable for conference/journal submission.
"""
import os, csv, glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, spearmanr

OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
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
    'savefig.format': 'png',
})

DATASET_NAMES = ['cviq', 'jufe', 'live', 'odi']
DATASET_LABELS = ['CVIQ', 'JUFE', 'LIVE 3D', 'ODI']
COLORS = plt.cm.Set2(np.linspace(0, 1, 4))
NUM_FOLDS = 5

OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)


# ── Training log loader ────────────────────────────────────────────────
def load_training_logs(dataset):
    logs = {}
    for fold in range(NUM_FOLDS):
        path = f'{dataset}_fold{fold}_training_logs.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Detect restarts and take the longest contiguous block
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


# ── 1. Cross-Dataset Heatmap ───────────────────────────────────────────
def plot_heatmap(df, metric='PCC', fname='heatmap_pcc.pdf'):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mat = df.pivot(index='trained_on', columns='evaluate_on', values=metric)
    mat = mat[DATASET_NAMES].reindex(DATASET_NAMES)
    vals = mat.values.astype(float)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect='equal')

    for i in range(4):
        for j in range(4):
            v = vals[i, j]
            color = 'white' if v > 0.65 else 'black'
            ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(DATASET_LABELS, rotation=30, ha='right')
    ax.set_yticklabels(DATASET_LABELS)
    ax.set_xlabel('Evaluation Dataset', fontsize=11)
    ax.set_ylabel('Training Dataset', fontsize=11)

    for i in range(4):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1,
                                fill=False, edgecolor='red', lw=2.5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric, fontsize=10)

    # Add grey "OOF" annotation on diagonal
    for i in range(4):
        ax.text(i, i, ' (OOF)', ha='left', va='center',
                fontsize=7, color='darkred', fontweight='bold',
                transform_rotates_text=False)

    title = f'Cross-Dataset {metric} (4 $\times$ 4 Matrix)'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 2. Combined PCC/SRCC Heatmap ──────────────────────────────────────
def plot_dual_heatmap(df, fname='heatmap_combined.pdf'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2),
                                    gridspec_kw={'wspace': 0.3})

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
                color = 'white' if v > 0.65 else 'black'
                ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(DATASET_LABELS, rotation=30, ha='right')
        ax.set_yticklabels(DATASET_LABELS)
        ax.set_xlabel('Evaluation Dataset')
        ax.set_ylabel('Training Dataset')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        for i in range(4):
            ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1,
                                    fill=False, edgecolor='red', lw=2))

    fig.colorbar(im, ax=[ax1, ax2], fraction=0.02, pad=0.02,
                 label='Correlation')
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 3. Scatter Plots (OOF predictions vs ground truth) ─────────────────
def plot_scatters(fname='scatter_oof.pdf'):
    kfold_files = {
        'LIVE 3D': 'live_kfold_predictions.csv',
        'ODI': 'odi_kfold_predictions.csv',
    }
    existing = {k: v for k, v in kfold_files.items() if os.path.exists(v)}
    if not existing:
        print("  No kfold prediction CSVs found, skipping scatter plots")
        return

    n = len(existing)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, existing.items()):
        df = pd.read_csv(path)
        if 'pred_raw' in df.columns:
            pred_col = 'pred_raw'
            gt_col = 'gt_quality'
        elif 'pred_z' in df.columns:
            pred_col = 'pred_z'
            gt_col = 'gt_norm'
        else:
            continue

        # Deduplicate by image_name (take first fold's prediction for OOF)
        oof = df.drop_duplicates(subset='image_name')
        preds = oof[pred_col].values.astype(float)
        gt = oof[gt_col].values.astype(float)

        pcc = pearsonr(preds, gt)[0]
        srcc = spearmanr(preds, gt)[0]

        ax.scatter(gt, preds, alpha=0.4, s=12, color=COLORS[0], edgecolors='none')
        lims = [min(gt.min(), preds.min()), max(gt.max(), preds.max())]
        ax.plot(lims, lims, 'r--', lw=1, alpha=0.6)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{label}\nPCC={pcc:.4f}  SRCC={srcc:.4f}',
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 4. Per-Fold OOF Bar Chart (for datasets with predictions) ─────────
def plot_fold_oof_bars(fname='fold_oof_bars.pdf'):
    datasets_with_folds = {}
    for ds in DATASET_NAMES:
        pred_files = glob.glob(f'{ds}_fold*_predictions.csv')
        if pred_files:
            # Load per-fold predictions
            fold_data = {}
            for pf in sorted(pred_files):
                fold = int(pf.split('_fold')[1].split('_')[0])
                df = pd.read_csv(pf)
                fold_data[fold] = df

            # Compute per-fold PCC/SRCC from pred_raw vs gt_quality
            pccs, srccs = [], []
            for fold in sorted(fold_data.keys()):
                df = fold_data[fold]
                preds = df['pred_raw'].values.astype(float)
                gt = df['gt_quality'].values.astype(float)
                pccs.append(pearsonr(preds, gt)[0])
                srccs.append(spearmanr(preds, gt)[0])
            datasets_with_folds[ds] = (pccs, srccs)

    if not datasets_with_folds:
        print("  No per-fold prediction CSVs found, skipping bar chart")
        return

    n = len(datasets_with_folds)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4))
    if n == 1:
        axes = [axes]

    colors_pcc = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    colors_srcc = plt.cm.Oranges(np.linspace(0.4, 0.9, 5))
    x = np.arange(5)
    w = 0.35

    for ax, (ds, (pccs, srccs)) in zip(axes, datasets_with_folds.items()):
        ax.bar(x - w/2, pccs, w, label='PCC', color=colors_pcc, edgecolor='k', lw=0.5)
        ax.bar(x + w/2, srccs, w, label='SRCC', color=colors_srcc, edgecolor='k', lw=0.5)
        ax.axhline(np.mean(pccs), color='steelblue', ls='--', lw=1,
                   label=f'μ PCC={np.mean(pccs):.4f}')
        ax.axhline(np.mean(srccs), color='darkorange', ls=':', lw=1,
                   label=f'μ SRCC={np.mean(srccs):.4f}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i}' for i in range(5)])
        ax.set_ylim(0, 1)
        ax.set_title(f'{ds.upper()} Per-Fold OOF',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Correlation')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 5. Cross-Dataset Transfer Bar Chart ────────────────────────────────
def plot_transfer_bars(df, fname='transfer_bars.pdf'):
    pcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    pcc_mat = pcc_mat[DATASET_NAMES].reindex(DATASET_NAMES)
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[DATASET_NAMES].reindex(DATASET_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, mat, metric in zip(axes, [pcc_mat, srcc_mat], ['PCC', 'SRCC']):
        x = np.arange(4)
        w = 0.18
        for i, (_, row) in enumerate(mat.iterrows()):
            offset = (i - 1.5) * w
            ax.bar(x + offset, row.values, w, label=row.name.upper(),
                   color=COLORS[i], edgecolor='k', lw=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_LABELS, rotation=20, ha='right')
        ax.set_ylabel(metric)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_title(f'{metric} by Training Dataset', fontsize=11, fontweight='bold')
        ax.legend(title='Train Set', fontsize=7, title_fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 6. LaTeX Table ─────────────────────────────────────────────────────
def save_latex_table(df, fname='table_cross_dataset.tex'):
    for metric, suffix in [('PCC', 'pcc'), ('SRCC', 'srcc')]:
        mat = df.pivot(index='trained_on', columns='evaluate_on', values=metric)
        mat = mat[DATASET_NAMES].reindex(DATASET_NAMES)
        lines = [
            r'\begin{tabular}{l' + 'c' * 4 + '}',
            r'\toprule',
            r'Training Set & ' + ' & '.join(DATASET_LABELS) + r' \\',
            r'\midrule',
        ]
        for i, ds in enumerate(DATASET_NAMES):
            row_vals = [f'{v:.4f}' for v in mat.loc[ds].values]
            row_str = f'{ds.upper()} & ' + ' & '.join(row_vals) + r' \\'
            lines.append(row_str)
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')

        tex = '\n'.join(lines)
        path = os.path.join(OUT_DIR, f'{suffix}_{fname}')
        with open(path, 'w') as f:
            f.write(tex)
        print(f"  Saved {path}")


# ── 7. Training Curves (Val PCC over epochs) ────────────────────────────
def plot_training_curves(fname='training_curves.pdf'):
    n_metrics = 3
    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 4.2),
                              gridspec_kw={'wspace': 0.35})
    metrics = [
        ('Val PCC', 'Val PCC', 'Validation PCC'),
        ('Val SRCC', 'Val SRCC', 'Validation SRCC'),
        ('Val Loss', 'Val Loss', 'Validation Loss'),
    ]

    for ax, (col, _, label) in zip(axes, metrics):
        for ds, color in zip(DATASET_NAMES, COLORS):
            logs = load_training_logs(ds)
            if not logs:
                continue
            # Interpolate to common epoch grid
            max_epochs = max(df['Epoch'].max() for df in logs.values())
            epochs = np.arange(1, max_epochs + 1)
            all_vals = []
            for fold in sorted(logs.keys()):
                df = logs[fold]
                vals = np.interp(epochs, df['Epoch'].values, df[col].values)
                all_vals.append(vals)
            mean_vals = np.mean(all_vals, axis=0)
            std_vals = np.std(all_vals, axis=0)
            ax.plot(epochs, mean_vals, color=color, label=ds.upper(), lw=1.5)
            ax.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals,
                            color=color, alpha=0.15)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if 'Loss' in label:
            ax.set_ylim(bottom=0)

    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 8. Best Performance Summary (Bar chart) ────────────────────────────
def plot_best_performance_summary(fname='best_perf_summary.pdf'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    x = np.arange(len(DATASET_NAMES))
    w = 0.3

    for ax, metric, label in zip([ax1, ax2],
                                  ['Val PCC', 'Val SRCC'],
                                  ['Val PCC', 'Val SRCC']):
        means, stds = [], []
        for ds in DATASET_NAMES:
            logs = load_training_logs(ds)
            _, _, pccs, srccs = get_best_per_fold(logs)
            if metric == 'Val PCC':
                vals = pccs
            else:
                vals = srccs
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.bar(x, means, w, yerr=stds, capsize=4, color=COLORS, edgecolor='k', lw=0.5)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02, f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_LABELS, rotation=20, ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)

    ax1.set_title('(a) Validation PCC', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Validation SRCC', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 9. Convergence Speed ───────────────────────────────────────────────
def plot_convergence(fname='convergence.pdf'):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(DATASET_NAMES))
    w = 0.3

    all_best_epochs = []
    for ds in DATASET_NAMES:
        logs = load_training_logs(ds)
        best_epochs, _, _, _ = get_best_per_fold(logs)
        all_best_epochs.append(best_epochs)

    means = [np.mean(be) for be in all_best_epochs]
    stds = [np.std(be) for be in all_best_epochs]

    ax.bar(x, means, w, yerr=stds, capsize=4, color=COLORS, edgecolor='k', lw=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1, f'{m:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_LABELS, rotation=20, ha='right')
    ax.set_ylabel('Epoch to Best Validation')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── 10. Combined Overview Dashboard ────────────────────────────────────
def plot_dashboard(df, fname='dashboard.pdf'):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # (a) PCC heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    mat = df.pivot(index='trained_on', columns='evaluate_on', values='PCC')
    mat = mat[DATASET_NAMES].reindex(DATASET_NAMES)
    vals = mat.values.astype(float)
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im = ax1.imshow(vals, cmap=cmap, norm=norm, aspect='equal')
    for i in range(4):
        for j in range(4):
            v = vals[i, j]
            ax1.text(j, i, f'{v:.4f}', ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='white' if v > 0.65 else 'black')
        ax1.add_patch(Rectangle((i-0.5, i-0.5), 1, 1,
                                fill=False, edgecolor='red', lw=2))
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(DATASET_LABELS, rotation=30, ha='right', fontsize=8)
    ax1.set_yticklabels(DATASET_LABELS, fontsize=8)
    ax1.set_title('(a) PCC Cross-Dataset Matrix', fontsize=10, fontweight='bold')

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
            vals = np.interp(epochs, logs[fold]['Epoch'].values, logs[fold]['Val EMA (3-ep)'].values)
            all_vals.append(vals)
        mean_vals = np.mean(all_vals, axis=0)
        std_vals = np.std(all_vals, axis=0)
        ax2.plot(epochs, mean_vals, color=color, label=ds.upper(), lw=1.5)
        ax2.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals,
                         color=color, alpha=0.12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val EMA')
    ax2.set_title('(b) Validation Training Curves', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # (c) Best performance bar
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(DATASET_NAMES))
    means_pcc, stds_pcc = [], []
    means_srcc, stds_srcc = [], []
    for ds in DATASET_NAMES:
        logs = load_training_logs(ds)
        _, _, pccs, srccs = get_best_per_fold(logs)
        means_pcc.append(np.mean(pccs))
        stds_pcc.append(np.std(pccs))
        means_srcc.append(np.mean(srccs))
        stds_srcc.append(np.std(srccs))
    w = 0.3
    ax3.bar(x - w/2, means_pcc, w, yerr=stds_pcc, capsize=3,
            color='steelblue', edgecolor='k', lw=0.5, label='PCC')
    ax3.bar(x + w/2, means_srcc, w, yerr=stds_srcc, capsize=3,
            color='darkorange', edgecolor='k', lw=0.5, label='SRCC')
    ax3.set_xticks(x)
    ax3.set_xticklabels(DATASET_LABELS, fontsize=8)
    ax3.set_ylabel('Best Val Correlation')
    ax3.set_title('(c) In-Dataset (OOF) Performance', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.grid(True, axis='y', alpha=0.3)

    # (d) Cross-dataset transfer bar
    ax4 = fig.add_subplot(gs[1, 1:])
    srcc_mat = df.pivot(index='trained_on', columns='evaluate_on', values='SRCC')
    srcc_mat = srcc_mat[DATASET_NAMES].reindex(DATASET_NAMES)
    x = np.arange(4)
    w = 0.18
    for i, (_, row) in enumerate(srcc_mat.iterrows()):
        offset = (i - 1.5) * w
        ax4.bar(x + offset, row.values, w, label=row.name.upper(),
                color=COLORS[i], edgecolor='k', lw=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(DATASET_LABELS, fontsize=8)
    ax4.set_ylabel('SRCC')
    ax4.set_title('(d) Cross-Dataset SRCC Breakdown', fontsize=10, fontweight='bold')
    ax4.legend(title='Train Set', fontsize=7, title_fontsize=8)
    ax4.grid(True, axis='y', alpha=0.3)

    # (e) Convergence
    ax5 = fig.add_subplot(gs[2, 0])
    all_best_epochs = []
    for ds in DATASET_NAMES:
        logs = load_training_logs(ds)
        best_epochs, _, _, _ = get_best_per_fold(logs)
        all_best_epochs.append(best_epochs)
    means_ep = [np.mean(be) for be in all_best_epochs]
    stds_ep = [np.std(be) for be in all_best_epochs]
    ax5.bar(x, means_ep, 0.4, yerr=stds_ep, capsize=3,
            color=COLORS, edgecolor='k', lw=0.5)
    for i, (m, s) in enumerate(zip(means_ep, stds_ep)):
        ax5.text(i, m + s + 1, f'{m:.0f}', ha='center', fontsize=8, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(DATASET_LABELS, fontsize=8)
    ax5.set_ylabel('Epochs')
    ax5.set_title('(e) Convergence Speed', fontsize=10, fontweight='bold')
    ax5.grid(True, axis='y', alpha=0.3)

    # (f) Scatter for LIVE
    ax6 = fig.add_subplot(gs[2, 1])
    if os.path.exists('live_kfold_predictions.csv'):
        df_scatter = pd.read_csv('live_kfold_predictions.csv')
        oof = df_scatter.drop_duplicates(subset='image_name')
        preds = oof['pred_raw'].values.astype(float)
        gt = oof['gt_quality'].values.astype(float)
        pcc = pearsonr(preds, gt)[0]
        srcc = spearmanr(preds, gt)[0]
        ax6.scatter(gt, preds, alpha=0.3, s=8, color=COLORS[2], edgecolors='none')
        lims = [min(gt.min(), preds.min()), max(gt.max(), preds.max())]
        ax6.plot(lims, lims, 'r--', lw=1, alpha=0.5)
        ax6.set_xlim(lims); ax6.set_ylim(lims)
        ax6.set_xlabel('GT'); ax6.set_ylabel('Pred')
        ax6.set_title(f'(f) LIVE 3D OOF\nPCC={pcc:.4f} SRCC={srcc:.4f}',
                      fontsize=9, fontweight='bold')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)

    # (g) Scatter for ODI
    ax7 = fig.add_subplot(gs[2, 2])
    if os.path.exists('odi_kfold_predictions.csv'):
        df_scatter = pd.read_csv('odi_kfold_predictions.csv')
        oof = df_scatter.drop_duplicates(subset='image_name')
        preds = oof['pred_raw'].values.astype(float)
        gt = oof['gt_quality'].values.astype(float)
        pcc = pearsonr(preds, gt)[0]
        srcc = spearmanr(preds, gt)[0]
        ax7.scatter(gt, preds, alpha=0.3, s=8, color=COLORS[3], edgecolors='none')
        lims = [min(gt.min(), preds.min()), max(gt.max(), preds.max())]
        ax7.plot(lims, lims, 'r--', lw=1, alpha=0.5)
        ax7.set_xlim(lims); ax7.set_ylim(lims)
        ax7.set_xlabel('GT'); ax7.set_ylabel('Pred')
        ax7.set_title(f'(g) ODI OOF\nPCC={pcc:.4f} SRCC={srcc:.4f}',
                      fontsize=9, fontweight='bold')
        ax7.set_aspect('equal')
        ax7.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    csv_path = 'cross_validate_all_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run cross_validate_all.py first.")
        return

    df = pd.read_csv(csv_path)
    print("Generating figures...")

    plot_heatmap(df, 'PCC', 'heatmap_pcc.png')
    plot_heatmap(df, 'SRCC', 'heatmap_srcc.png')
    plot_dual_heatmap(df, 'heatmap_combined.png')
    plot_scatters('scatter_oof.png')
    plot_fold_oof_bars('fold_oof_bars.png')
    plot_transfer_bars(df, 'transfer_bars.png')
    plot_training_curves('training_curves.png')
    plot_best_performance_summary('best_perf_summary.png')
    plot_convergence('convergence.png')
    plot_dashboard(df, 'dashboard.png')
    save_latex_table(df, 'cross_dataset.tex')

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
