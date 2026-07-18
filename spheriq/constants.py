"""
Shared constants used across the training pipeline.
Do not define these values in individual modules — import from here.
"""
import torch

# ERP distortion weight per cubemap face.
# Derived from the solid-angle integral over each cubemap face.
# Order: 0:+X(right)  1:-X(left)  2:+Y(top)  3:-Y(bottom)  4:+Z(front)  5:-Z(back)
# Equatorial faces weight 1.0; polar faces weight 0.552.
_RAW_ERP_WEIGHTS = torch.tensor([1.0, 1.0, 0.552, 0.552, 1.0, 1.0])
ERP_FACE_WEIGHTS = _RAW_ERP_WEIGHTS / _RAW_ERP_WEIGHTS.sum()   # sums to 1.0
