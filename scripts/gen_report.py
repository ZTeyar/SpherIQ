#!/usr/bin/env python3
"""Generate comprehensive evaluation report as Markdown."""

import csv
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def group_by(lst, key_fn):
    d = defaultdict(list)
    for item in lst:
        d[key_fn(item)].append(item)
    return dict(d)


def compute_metrics(preds, gts):
    pcc = pearsonr(preds, gts)[0]
    srcc = spearmanr(preds, gts)[0]
    return pcc, srcc


# ── LIVE ──────────────────────────────────────────────────────────────────────
def process_live():
    rows = read_csv("live_kfold_predictions.csv")
    for r in rows:
        parts = r["image_name"].split("_")
        r["scene"] = parts[0]
        r["distortion"] = parts[1]
        r["level"] = parts[2] if len(parts) > 2 else "0"

    lines = []
    def L(text=""):
        lines.append(text)

    L("# Evaluation Results")
    L()
    L("## LIVE 3D IQA (LIVE-trained models, TTA yaw=0)")
    L()

    # ── Per-fold ─────────────────────────────────────────────────────────
    L("### Per-Fold Results")
    L()
    live_per_fold = [
        (0, 75, 90, 0.7660, 0.7883),
        (1, 82, 90, 0.8709, 0.8908),
        (2, 37, 89, 0.7750, 0.8301),
        (3, 99, 90, 0.7546, 0.8112),
        (4, 76, 90, 0.8531, 0.8933),
    ]
    logged = [
        (0.7674, 0.7860, 0.7767),
        (0.8714, 0.8908, 0.8811),
        (0.7733, 0.8317, 0.8025),
        (0.7522, 0.8146, 0.7834),
        (0.8529, 0.8929, 0.8729),
    ]
    L()
    L("| Fold | Best Epoch | Samples | PCC | SRCC | Logged PCC | Logged SRCC | Logged Combined |")
    L("|------|-----------|--------|------|------|------------|-------------|-----------------|")
    pccs, srccs = [], []
    for (f, ep, n, p, s), (lp, ls, lc) in zip(live_per_fold, logged):
        L(f"| {f} | {ep} | {n} | {p:.4f} | {s:.4f} | {lp:.4f} | {ls:.4f} | {lc:.4f} |")
        pccs.append(p)
        srccs.append(s)
    L(f"| **Mean** | | | **{np.mean(pccs):.4f}** | **{np.mean(srccs):.4f}** | | | |")
    L(f"| **Std**  | | | **±{np.std(pccs, ddof=1):.4f}** | **±{np.std(srccs, ddof=1):.4f}** | | | |")
    L()

    # ── Per-scene ─────────────────────────────────────────────────────────
    L("### Per-Scene Results")
    L()
    L("Each scene is held out in exactly one fold. Metrics computed using that fold's predictions.")
    L()
    L("| Scene | Fold | Samples | PCC | SRCC |")
    L("|-------|------|---------|------|------|")

    fold_data = group_by(rows, lambda r: int(r["fold"]))
    scene_rows = []
    for fold in sorted(fold_data):
        r = fold_data[fold]
        scene_groups = group_by(r, lambda x: x["scene"])
        for scene in sorted(scene_groups):
            gr = scene_groups[scene]
            preds = np.array([float(x["pred_raw"]) for x in gr])
            gts = np.array([float(x["gt_quality"]) for x in gr])
            p, s = compute_metrics(preds, gts)
            scene_rows.append((scene, fold, p, s, len(gr)))

    scene_rows.sort(key=lambda x: (x[1], x[0]))
    scene_pccs, scene_srccs = [], []
    for scene, fold, p, s, n in scene_rows:
        L(f"| {scene} | {fold} | {n} | {p:.4f} | {s:.4f} |")
        scene_pccs.append(p)
        scene_srccs.append(s)
    L(f"| **Mean** | | | **{np.mean(scene_pccs):.4f}** | **{np.mean(scene_srccs):.4f}** |")
    L(f"| **Std**  | | | **±{np.std(scene_pccs, ddof=1):.4f}** | **±{np.std(scene_srccs, ddof=1):.4f}** |")
    L()

    # ── Per-distortion ────────────────────────────────────────────────────
    L("### Per-Distortion-Type Results")
    L()
    L("Aggregated across all folds (each image predicted by its held-out fold model).")
    L()
    L("| Distortion | Samples | PCC | SRCC |")
    L("|-----------|---------|------|------|")

    dist_data = group_by(rows, lambda r: r["distortion"])
    for dist in sorted(dist_data):
        gr = dist_data[dist]
        preds = np.array([float(x["pred_raw"]) for x in gr])
        gts = np.array([float(x["gt_quality"]) for x in gr])
        p, s = compute_metrics(preds, gts)
        L(f"| {dist} | {len(gr)} | {p:.4f} | {s:.4f} |")
    L()

    # ── Ensemble ──────────────────────────────────────────────────────────
    L("### Ensemble (All 449 predictions)")
    all_preds = np.array([float(r["pred_raw"]) for r in rows])
    all_gts = np.array([float(r["gt_quality"]) for r in rows])
    ep, es = compute_metrics(all_preds, all_gts)
    L(f"- **Linear PCC:** {ep:.4f}")
    L(f"- **SRCC:** {es:.4f}")
    L()

    return lines


# ── ODI ───────────────────────────────────────────────────────────────────────
def process_odi():
    rows = read_csv("odi_kfold_predictions.csv")
    for r in rows:
        parts = r["image_name"].split("_")
        r["qf_level"] = parts[0]
        r["projection"] = parts[1]
        r["scene_type"] = parts[2]
        r["pid"] = parts[3]

    lines = []
    def L(text=""):
        lines.append(text)

    L("## ODI (Zero-shot: LIVE-trained models → ODI)")
    L()

    # ── Per-fold ─────────────────────────────────────────────────────────
    L("### Per-Fold Results")
    L()
    L("| Fold | Samples | PCC | SRCC |")
    L("|------|---------|------|------|")

    fold_data = group_by(rows, lambda r: int(r["fold"]))
    fold_pccs, fold_srccs = [], []
    for fold in sorted(fold_data):
        gr = fold_data[fold]
        preds = np.array([float(x["pred_raw"]) for x in gr])
        gts = np.array([float(x["gt_quality"]) for x in gr])
        p, s = compute_metrics(preds, gts)
        fold_pccs.append(p)
        fold_srccs.append(s)
        L(f"| {fold} | {len(gr)} | {p:.4f} | {s:.4f} |")
    L(f"| **Mean** | | **{np.mean(fold_pccs):.4f}** | **{np.mean(fold_srccs):.4f}** |")
    L(f"| **Std**  | | **±{np.std(fold_pccs, ddof=1):.4f}** | **±{np.std(fold_srccs, ddof=1):.4f}** |")
    L()

    # ── Per scene type ───────────────────────────────────────────────────
    L("### Per-Scene-Type Results")
    L()
    L("| Scene Type | Samples | PCC | SRCC |")
    L("|-----------|---------|------|------|")

    scene_data = group_by(rows, lambda r: r["scene_type"])
    for st in sorted(scene_data):
        gr = scene_data[st]
        preds = np.array([float(x["pred_raw"]) for x in gr])
        gts = np.array([float(x["gt_quality"]) for x in gr])
        p, s = compute_metrics(preds, gts)
        L(f"| {st} | {len(gr)} | {p:.4f} | {s:.4f} |")
    L()

    # ── Per projection mode ──────────────────────────────────────────────
    L("### Per-Projection-Mode Results")
    L()
    L("| Projection | Samples | PCC | SRCC |")
    L("|-----------|---------|------|------|")

    proj_data = group_by(rows, lambda r: r["projection"])
    for proj in sorted(proj_data):
        gr = proj_data[proj]
        preds = np.array([float(x["pred_raw"]) for x in gr])
        gts = np.array([float(x["gt_quality"]) for x in gr])
        p, s = compute_metrics(preds, gts)
        L(f"| {proj} | {len(gr)} | {p:.4f} | {s:.4f} |")
    L()

    # ── Per QF level ─────────────────────────────────────────────────────
    L("### Per-QF-Level Results")
    L()
    L("| QF Level | Samples | PCC | SRCC |")
    L("|---------|---------|------|------|")

    qf_data = group_by(rows, lambda r: r["qf_level"])
    for qf in sorted(qf_data):
        gr = qf_data[qf]
        preds = np.array([float(x["pred_raw"]) for x in gr])
        gts = np.array([float(x["gt_quality"]) for x in gr])
        p, s = compute_metrics(preds, gts)
        L(f"| {qf} | {len(gr)} | {p:.4f} | {s:.4f} |")
    L()

    # ── Ensemble ─────────────────────────────────────────────────────────
    L("### Ensemble (Average of 5 Folds)")
    L()

    img_data = defaultdict(list)
    for r in rows:
        img_data[r["image_name"]].append(float(r["pred_raw"]))

    img_names = sorted(img_data)
    ensemble_preds = np.array([np.mean(img_data[n]) for n in img_names])
    gt_map = {}
    for r in rows:
        if r["image_name"] not in gt_map:
            gt_map[r["image_name"]] = float(r["gt_quality"])
    ensemble_gts = np.array([gt_map[n] for n in img_names])

    ep, es = compute_metrics(ensemble_preds, ensemble_gts)
    L(f"- **Linear PCC:** {ep:.4f}")
    L(f"- **SRCC:** {es:.4f}")
    L()

    from scipy.optimize import curve_fit
    def vqeg5pl(x, b1, b2, b3, b4, b5):
        return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5
    p0 = [np.ptp(ensemble_gts), 1.0, np.mean(ensemble_preds), 0.1, np.min(ensemble_gts)]
    try:
        popt, _ = curve_fit(vqeg5pl, ensemble_preds, ensemble_gts, p0=p0, maxfev=10000)
        mapped = vqeg5pl(ensemble_preds, *popt)
        nl_pcc = pearsonr(mapped, ensemble_gts)[0]
        L(f"- **Nonlinear PLCC (VQEG 5-param logistic):** {nl_pcc:.4f}")
    except RuntimeError:
        pass
    L()

    return lines


def main():
    live_lines = process_live()
    odi_lines = process_odi()

    all_lines = live_lines + odi_lines
    with open("EVALUATION_RESULTS.md", "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"Written EVALUATION_RESULTS.md ({len(all_lines)} lines)")


if __name__ == "__main__":
    main()
