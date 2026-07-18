import pandas as pd
import numpy as np
from typing import cast
from spheriq.utils import display_text


# ---------------------------------------------------------------------------
# Dataset registry — maps lowercase dataset name substrings to the correct
# is_dmos flag.  LIVE3D uses DMOS (lower raw = better quality, is_dmos=True).
# Add new datasets here to prevent silent sign inversion.
#
# IMPORTANT: The registry uses substring matching on dataset_name.lower().
# Keys should be as specific as possible to avoid false positives.
# "live" matches any dataset name containing "live", which is intentional
# for the LIVE 3D IQA dataset used in the notebook (dataset_name="live").
# ---------------------------------------------------------------------------
KNOWN_DATASET_SCORE_TYPES: dict[str, bool] = {
    'live3d':   True,   # DMOS scale (lower raw = better quality)
    'live_3d':  True,   # alternate spelling
    'live':     True,   # short form used in notebook
    'cviq':     False,  # MOS
    'oiqa':     False,  # MOS
    'odi':      True,   # DMOS (lower raw = better quality)
    'ivqad':    True,   # DMOS
}


def infer_is_dmos(dataset_name: str) -> bool | None:
    """
    Infer the is_dmos flag from the dataset name using the registry above.

    Returns None if the dataset is not in the registry — the caller must then
    pass is_dmos explicitly.  Raises ValueError if an explicit is_dmos
    disagrees with the registry, to catch copy-paste errors early.
    """
    lower = dataset_name.lower()
    for key, val in KNOWN_DATASET_SCORE_TYPES.items():
        if key in lower:
            return val
    return None


def process_labels(input_csv_path, output_csv_path, is_dmos: bool | None = None,
                   dataset_name: str = ''):
    """
    Process raw scores for training while keeping them true to ground truth.

    This version uses per-dataset min-max normalisation to map all scores
    strictly into a [0, 1] range (where 1 is the highest quality). This
    ensures perfect cross-dataset compatibility. It also preserves
    the original raw scores and the normalisation parameters.

    Args:
        input_csv_path:  Path to raw CSV with 'Image Name' and 'DMOS' (or 'MOS') columns.
        output_csv_path: Destination path for the processed CSV.
        is_dmos:         True if higher score means lower quality (DMOS).
                         False if higher score means higher quality (MOS).
                         Pass None to auto-detect from dataset_name (recommended).
        dataset_name:    Optional name used to auto-detect is_dmos from the registry
                         and to cross-check an explicitly passed is_dmos value.

    Output columns:
        image_name:      Cleaned image name.
        quality_score:   Min-Max scaled quality in [0, 1] (higher = better).
        raw_score:       The original ground truth value (true to source).
        dataset_mean:    Mean of raw scores.
        dataset_std:     Std of raw scores.
        dataset_min:     Min of raw scores.
        dataset_max:     Max of raw scores.

    CRITICAL — sign convention
    --------------------------
    Getting is_dmos wrong silently inverts every quality score.  A model
    trained on inverted scores learns that blurry images are high-quality and
    sharp images are low-quality.  The resulting SRCC will be near −0.7 instead
    of +0.7.  Always verify by checking that a known-good image in your dataset
    maps to a positive quality_score after processing.

    Known dataset flags (see KNOWN_DATASET_SCORE_TYPES above):
      LIVE3D  → is_dmos=True   (DMOS, lower raw = better quality)
      ODI     → is_dmos=True   (DMOS, lower raw = better quality)
    """
    # ── Resolve is_dmos ───────────────────────────────────────────────────────
    registry_val = infer_is_dmos(dataset_name) if dataset_name else None

    if is_dmos is None:
        if registry_val is None:
            raise ValueError(
                f"process_labels(): is_dmos was not provided and dataset_name "
                f"'{dataset_name}' is not in the registry (KNOWN_DATASET_SCORE_TYPES). "
                f"Pass is_dmos=True (DMOS) or is_dmos=False (MOS) explicitly."
            )
        is_dmos = registry_val
        display_text(
            f"[prepare_scores] Auto-detected is_dmos={is_dmos} for '{dataset_name}' "
            f"from registry. Verify this matches your dataset's score convention."
        )
    elif registry_val is not None and is_dmos != registry_val:
        raise ValueError(
            f"process_labels(): explicit is_dmos={is_dmos} contradicts the registry "
            f"value {registry_val} for dataset '{dataset_name}'. "
            f"If your dataset genuinely differs, add it to KNOWN_DATASET_SCORE_TYPES "
            f"or remove the dataset_name argument to suppress this check."
        )
    df = pd.read_csv(input_csv_path, header=0)

    # Detect score column
    score_col = 'DMOS' if 'DMOS' in df.columns else 'MOS'
    if score_col not in df.columns:
        # Fallback to any column that isn't 'Image Name'
        other_cols = [c for c in df.columns if c != 'Image Name']
        if not other_cols:
            raise ValueError(f"No score column found in {input_csv_path}")
        score_col = other_cols[0]

    raw_scores = df[score_col].astype(float).values
    
    # Calculate robust stats for this dataset (1st and 99th percentiles)
    # to prevent a single noisy label outlier from squashing the quality range.
    ds_mean = float(np.mean(raw_scores))
    ds_std  = float(np.std(raw_scores))
    ds_min  = float(np.percentile(raw_scores, 1))
    ds_max  = float(np.percentile(raw_scores, 99))
    if ds_std < 1e-8:
        ds_std = 1.0

    # Convert to quality (higher = better) and map to [0, 1]
    # If DMOS: quality = (max - raw) / (max - min)
    # If MOS:  quality = (raw - min) / (max - min)
    range_val = ds_max - ds_min
    if range_val < 1e-8:
        range_val = 1.0

    if is_dmos:
        quality_scores = (ds_max - raw_scores) / range_val
    else:
        quality_scores = (raw_scores - ds_min) / range_val
    
    # Clamp to [0, 1] since raw scores may lie outside the [1st, 99th] percentile range
    quality_scores = np.clip(quality_scores, 0.0, 1.0)

    import re
    image_names = (
        df['Image Name']
        .astype(str)
        .str.replace(r'\.(jpe?g|png|webp)$', '', regex=True, flags=re.IGNORECASE)
        .tolist()
    )

    new_df = pd.DataFrame({
        'image_name':    image_names,
        'quality_score': [f'{v:.6f}' for v in quality_scores],
        'raw_score':     [f'{v:.6f}' for v in raw_scores],
        'dataset_mean':  [f'{ds_mean:.6f}'] * len(image_names),
        'dataset_std':   [f'{ds_std:.6f}'] * len(image_names),
        'dataset_min':   [f'{ds_min:.6f}'] * len(image_names),
        'dataset_max':   [f'{ds_max:.6f}'] * len(image_names),
        'score_type':    ['DMOS' if is_dmos else 'MOS'] * len(image_names),
    })

    new_df.to_csv(output_csv_path, index=False)
    display_text(
        f"Processed {len(image_names)} scores → {output_csv_path}\n"
        f"Truth preserved in 'raw_score'. Training 'quality_score' is min-max scaled to [0, 1] (higher=better).\n"
        f"Dataset Stats: min={ds_min:.4f}, max={ds_max:.4f} ({'DMOS' if is_dmos else 'MOS'})"
    )
if __name__ == "__main__":
    process_labels(input_csv_path="LIVE_3D.csv", output_csv_path="live_scores.csv", is_dmos=True)
