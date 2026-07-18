import csv
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
import math

CSV_PATH = "live_kfold_predictions.csv"

def vqeg5pl(x, b1, b2, b3, b4, b5):
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))) + b4 * x + b5

def load_csv(path):
    folds = {}
    all_pred = []
    all_gt = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold = int(row["fold"])
            pred = float(row["pred_raw"])
            gt = float(row["gt_quality"])
            entry = {"image": row["image_name"], "pred": pred, "gt": gt}
            folds.setdefault(fold, []).append(entry)
            all_pred.append(pred)
            all_gt.append(gt)
    return folds, np.array(all_pred), np.array(all_gt)

def fit_and_evaluate(pred, gt):
    p0 = [np.ptp(gt), 1.0, np.mean(pred), 0.1, np.min(gt)]
    try:
        popt, _ = curve_fit(
            vqeg5pl, pred, gt, p0=p0, maxfev=10000,
            bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf, np.inf])
        )
    except RuntimeError as e:
        print(f"    curve_fit failed: {e}")
        popt = None
    if popt is not None:
        mapped = vqeg5pl(pred, *popt)
    else:
        mapped = pred
    plcc = pearsonr(mapped, gt)[0]
    srcc = spearmanr(pred, gt)[0]
    return plcc, srcc, popt

def main():
    folds_data, all_pred, all_gt = load_csv(CSV_PATH)

    print("=" * 65)
    print(f"{'Fold':<6} {'Samples':<8} {'Linear PLCC':<13} {'Nonlinear PLCC':<15} {'SRCC':<10} {'Improvement':<12}")
    print("=" * 65)

    fold_linear_plcc = []
    fold_nonlinear_plcc = []
    fold_srcc = []

    for fold in sorted(folds_data.keys()):
        entries = folds_data[fold]
        pred = np.array([e["pred"] for e in entries])
        gt = np.array([e["gt"] for e in entries])

        linear_plcc = pearsonr(pred, gt)[0]
        srcc = spearmanr(pred, gt)[0]
        nl_plcc, nl_srcc, popt = fit_and_evaluate(pred, gt)

        fold_linear_plcc.append(linear_plcc)
        fold_nonlinear_plcc.append(nl_plcc)
        fold_srcc.append(srcc)
        impr = nl_plcc - linear_plcc

        opt_str = ""
        if popt is not None:
            opt_str = f"  β={np.array2string(popt, precision=4, suppress_small=True)}"

        print(f"{fold:<6} {len(entries):<8} {linear_plcc:<13.4f} {nl_plcc:<15.4f} {srcc:<10.4f} {impr:<+12.4f}{opt_str}")

    # Overall
    linear_all = pearsonr(all_pred, all_gt)[0]
    srcc_all = spearmanr(all_pred, all_gt)[0]
    nl_all, nl_srcc_all, popt_all = fit_and_evaluate(all_pred, all_gt)
    impr_all = nl_all - linear_all
    print("=" * 65)
    opt_all_str = ""
    if popt_all is not None:
        opt_all_str = f"  β={np.array2string(popt_all, precision=4, suppress_small=True)}"
    print(f"{'All':<6} {len(all_pred):<8} {linear_all:<13.4f} {nl_all:<15.4f} {srcc_all:<10.4f} {impr_all:<+12.4f}{opt_all_str}")
    print("=" * 65)

    # Mean ± std
    linear_mean = np.mean(fold_linear_plcc)
    linear_std = np.std(fold_linear_plcc, ddof=1)
    nl_mean = np.mean(fold_nonlinear_plcc)
    nl_std = np.std(fold_nonlinear_plcc, ddof=1)
    srcc_mean = np.mean(fold_srcc)
    srcc_std = np.std(fold_srcc, ddof=1)
    impr_mean = nl_mean - linear_mean
    print(f"{'Mean':<6} {'':<8} {linear_mean:<13.4f} {nl_mean:<15.4f} {srcc_mean:<10.4f} {impr_mean:<+12.4f}")
    print(f"{'Std':<6} {'':<8} {linear_std:<13.4f} {nl_std:<15.4f} {srcc_std:<10.4f} {'':>12}")

if __name__ == "__main__":
    main()
