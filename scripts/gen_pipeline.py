#!/usr/bin/env python3
"""SpherIQ detailed pipeline diagram — drawsvg with dimensions, losses, training flow."""

import drawsvg as dw
import os, math

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────
C_INPUT  = '#2E7D32'
C_AUG    = '#1565C0'
C_PRE    = '#E65100'
C_EMB    = '#6A1B9A'
C_ENC    = '#283593'
C_FUSE   = '#4E342E'
C_PRED   = '#AD1457'
C_INFER  = '#00838F'
C_OUT    = '#2E7D32'
C_LOSS   = '#B71C1C'
C_TRAIN  = '#F57F17'
C_DIM    = '#546E7A'

W = 880
CX = 420      # centre for main pipeline
DIM_X = 20    # left-side dimension annotations
ANN_R = 660   # right-side annotations
Y_OFF = 90

d = dw.Drawing(W, 1750)

def tag(dwg, x, y, text, color=C_DIM, size=7.5):
    dwg.append(dw.Text(text, size, x, y, fill=color, font_family='monospace'))

def box(dwg, cx, y, w, h, color, lines, sub=None, dim=None, ann=None):
    """Draw a centred rounded box with multiple bold lines + optional sub and side notes."""
    x = cx - w/2
    dwg.append(dw.Rectangle(x, y, w, h, fill=color+'15', stroke=color,
                            stroke_width=2, rx=7, ry=7))
    n = len(lines)
    total_h = n * 12 - 4
    start_y = y + (h - total_h) / 2 + 8
    for i, line in enumerate(lines):
        is_bold = not line.startswith(' ')
        dwg.append(dw.Text(line.strip(), 9 if is_bold else 7.5, cx, start_y + i*13,
                           fill='#222' if is_bold else '#555',
                           font_weight='bold' if is_bold else 'normal',
                           font_family='sans-serif', text_anchor='middle'))
    if sub:
        dwg.append(dw.Text(sub, 7, cx, y+h-5, fill='#777',
                           font_family='sans-serif', text_anchor='middle'))
    if dim:
        tag(dwg, DIM_X, y+h//2+3, dim)
    if ann:
        tag(dwg, ANN_R, y+h//2+3, ann, color='#888', size=7)

def arrow(dwg, y):
    dwg.append(dw.Lines(CX, y-30, CX, y-8, stroke='#888', stroke_width=2,
                        fill='none', arrow_end=True))

def arrow_label(dwg, y, label, color='#546E7A'):
    dwg.append(dw.Text(label, 7.5, CX, y-14, fill=color, font_family='monospace',
                       text_anchor='middle', font_weight='bold'))

# ═══════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════
d.append(dw.Text('SpherIQ Pipeline', 22, CX, 32, fill='#222',
                 font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('Omnidirectional Image Quality Assessment  —  Detailed Architecture & Data Flow', 11, CX, 50,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

# Section: Input + Augmentation header
d.append(dw.Text('DATA FLOW  ●──→  TENSOR DIMENSIONS', 8, CX, 72,
                 fill='#999', font_family='monospace', text_anchor='middle'))

y = Y_OFF

# ═══ 1 — ERP INPUT ══════════════════════════════════════════════════
box(d, CX, y, 240, 50, C_INPUT,
    ['ERP 360° Image', 'Equirectangular projection'],
    dim='(B, 3, H, W)   H=2W')

y += 72
arrow(d, y)
arrow_label(d, y, 'augment?  yes ──► (training)     no ──► (eval: skip to cubemap)')

# ═══ 2 — DATA AUGMENTATION ══════════════════════════════════════════
aug_y = y
# Group box for augmentation
gw = 640; gh = 120
gx = CX - gw/2
d.append(dw.Rectangle(gx, y, gw, gh, fill=C_AUG+'08', stroke=C_AUG+'40',
                      stroke_width=1.5, rx=10, ry=10, stroke_dasharray='6,3'))
d.append(dw.Text('Data Augmentation  (training only)', 10, gx+12, y+18,
                 fill=C_AUG, font_weight='bold', font_family='sans-serif'))

# Sub-boxes within
bw = 190; bh = 72
by = y + 28
d.append(dw.Rectangle(gx+14, by, bw, bh, fill='white', stroke=C_AUG, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Spherical 3D Rotation', 9, gx+14+bw/2, by+24,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  yaw: uniform [0°,360°) | N(0°,60°)', 7, gx+14+bw/2, by+40,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  pitch: U[-25°,25°]   roll: U[-18°,18°]', 7, gx+14+bw/2, by+54,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  → 3D rotation matrix R ∈ SO(3)', 7, gx+14+bw/2, by+66,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

# Arrow right
d.append(dw.Lines(gx+14+bw+4, by+36, gx+14+bw+20, by+36,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+bw+24, by, bw, bh, fill='white', stroke=C_AUG, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Fused Single-Pass', 9, gx+14+bw+24+bw/2, by+24,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('Cubemap Projection', 9, gx+14+bw+24+bw/2, by+38,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  compose R with cube-face grids', 7, gx+14+bw+24+bw/2, by+54,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  sample original ERP once  ✓', 7, gx+14+bw+24+bw/2, by+66,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

d.append(dw.Lines(gx+14+2*bw+28, by+36, gx+14+2*bw+44, by+36,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+2*bw+48, by, bw, bh, fill='white', stroke=C_AUG, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Synthetic Artifacts', 9, gx+14+2*bw+48+bw/2, by+24,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  G. blur σ∈[0.5,2.0]', 7, gx+14+2*bw+48+bw/2, by+40,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  Gauss noise σ∈[0.01,0.08]', 7, gx+14+2*bw+48+bw/2, by+54,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('  downscale 0.4-0.9×', 7, gx+14+2*bw+48+bw/2, by+66,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

y += gh + 10
arrow(d, y)
arrow_label(d, y, 'Viewport face dropout (p=0.15, equatorial only)')

# ═══ 3 — CUBEMAP + PATCHES ══════════════════════════════════════════
cm_y = y
cgh = 130; cgw = 640
d.append(dw.Rectangle(gx, y, cgw, cgh, fill=C_PRE+'08', stroke=C_PRE+'40',
                      stroke_width=1.5, rx=10, ry=10, stroke_dasharray='6,3'))
d.append(dw.Text('Preprocessing & Patch Extraction', 10, gx+12, y+18,
                 fill=C_PRE, font_weight='bold', font_family='sans-serif'))

by = y + 28
d.append(dw.Rectangle(gx+14, by, 280, 90, fill='white', stroke=C_PRE, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Cubemap Projection', 9, gx+14+140, by+22,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('ERP → 6 cube faces: ±X, ±Y, ±Z', 7.5, gx+14+140, by+40,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('each face: 512×512 px', 7.5, gx+14+140, by+54,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('stereo: 12 faces (2 views × 6)', 7.5, gx+14+140, by+68,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('RGB, normalized to [-1, 1]', 7.5, gx+14+140, by+82,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

d.append(dw.Lines(gx+14+280+2, by+45, gx+14+280+18, by+45,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+280+22, by, 310, 90, fill='white', stroke=C_PRE, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Multi-Scale Patch Extraction', 9, gx+14+280+22+155, by+22,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('3 scales: longer-side lengths [224, 384, 512]', 7.5, gx+14+280+22+155, by+40,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('grid 10×10 per face per scale', 7.5, gx+14+280+22+155, by+54,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('patch size 32×32', 7.5, gx+14+280+22+155, by+68,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('~2700 tokens total per view', 7.5, gx+14+280+22+155, by+82,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

y += cgh + 10
arrow(d, y)
arrow_label(d, y, '(B, 6×N_patches, 3, 32, 32)')

# ═══ 4 — CNN BACKBONE ════════════════════════════════════════════════
box(d, CX, y, 520, 60, C_EMB,
    ['CNN Backbone  ──  Per-patch feature extraction'],
    [' StdConv(7×7, s2) → GN → MaxPool(3×3, s2) → Bottleneck → Linear → 384-d'])
y += 82
arrow(d, y)
arrow_label(d, y, '(B, N_total, 384)')

# ═══ 5 — GEOMETRIC EMBEDDINGS ════════════════════════════════════════
emb_y = y
eh = 100; ew = 640
d.append(dw.Rectangle(gx, y, ew, eh, fill=C_EMB+'08', stroke=C_EMB+'40',
                      stroke_width=1.5, rx=10, ry=10, stroke_dasharray='6,3'))
d.append(dw.Text('Geometric Embeddings', 10, gx+12, y+18,
                 fill=C_EMB, font_weight='bold', font_family='sans-serif'))

by = y + 28
bw3 = 190
d.append(dw.Rectangle(gx+14, by, bw3, 62, fill='white', stroke=C_EMB, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('3D Rotary PE (RoPE)', 9, gx+14+bw3/2, by+18,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('continuous unit-sphere coords', 7.5, gx+14+bw3/2, by+34,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('cube face → 3D mapping', 7.5, gx+14+bw3/2, by+48,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

d.append(dw.Lines(gx+14+bw3+4, by+31, gx+14+bw3+18, by+31,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+bw3+22, by, 195, 62, fill='white', stroke=C_EMB, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Face Embedding', 9, gx+14+bw3+22+97, by+18,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('6 learned vectors × 384-d', 7.5, gx+14+bw3+22+97, by+34,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('disambiguates cube faces', 7.5, gx+14+bw3+22+97, by+48,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

d.append(dw.Lines(gx+14+2*bw3+26, by+31, gx+14+2*bw3+40, by+31,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+2*bw3+44, by, 180, 62, fill='white', stroke=C_EMB, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Scale Embedding', 9, gx+14+2*bw3+44+90, by+18,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('3 learned vectors × 384-d', 7.5, gx+14+2*bw3+44+90, by+34,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('encodes multi-scale level', 7.5, gx+14+2*bw3+44+90, by+48,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

y += eh + 10
d.append(dw.Text('[CLS tokens: 6 × 384-d  (1 per face)]', 7.5, CX, y-2,
                 fill='#666', font_family='sans-serif', text_anchor='middle'))
arrow(d, y)
arrow_label(d, y, '(B, 6+N_patches, 384)  +  RoPE applied in attn')

# ═══ 6 — TRANSFORMER ENCODER ═════════════════════════════════════════
enc_y = y
d.append(dw.Rectangle(gx, y, ew, 130, fill=C_ENC+'08', stroke=C_ENC+'40',
                      stroke_width=1.5, rx=10, ry=10, stroke_dasharray='6,3'))
d.append(dw.Text('Transformer Encoder', 10, gx+12, y+18,
                 fill=C_ENC, font_weight='bold', font_family='sans-serif'))

by = y + 28
# Internal block diagram
ibw = 600
d.append(dw.Rectangle(gx+20, by, ibw, 88, fill='white', stroke=C_ENC, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Block ×14  (stochastic depth decay 0 → 0.1)', 8, gx+20+ibw/2, by+14,
                 fill='#333', font_family='sans-serif', text_anchor='middle'))
# LayerNorm → MHA → LayerNorm → MLP
for i, (label, sub_label, w_mod) in enumerate([
    ('LayerNorm', 'eps=1e-6', 85),
    ('Multi-Head\nAttention', '4 heads, d=384', 120),
    ('DropPath', 'stoch. depth', 85),
    ('LayerNorm', 'eps=1e-6', 85),
    ('MLP', '1152→384, GELU', 100),
    ('DropPath', 'stoch. depth', 85),
]):
    mx = gx+30 + i*97 + (i>0)*(sum([85,120,85,85,100,85][:i])-i*97)
    # I'll simplify: just draw the block description
    pass

d.append(dw.Text('  LayerNorm → Multi-Head Self-Attention (4 heads) → DropPath →', 8, gx+20+ibw/2, by+40,
                 fill='#444', font_family='monospace', text_anchor='middle'))
d.append(dw.Text('  LayerNorm → MLP (384→1152→384, GELU) → DropPath', 8, gx+20+ibw/2, by+56,
                 fill='#444', font_family='monospace', text_anchor='middle'))
d.append(dw.Text('  dropout=0.2  attn_dropout=0.1  stochastic depth=0.1', 7.5, gx+20+ibw/2, by+76,
                 fill='#666', font_family='sans-serif', text_anchor='middle'))

y += 130 + 10
arrow(d, y)
arrow_label(d, y, '(B, 6+N_patches, 384)  —  extract CLS tokens')

# ═══ 7 — META-TRANSFORMER ════════════════════════════════════════════
box(d, CX, y, 520, 55, C_FUSE,
    ['Meta-Transformer  (Face Aggregation)'],
    ['3 layers, 6 heads · ERP-weighted attn bias (7×7 learned + per-face column bias)'])
y += 77
arrow(d, y)

d.append(dw.Text('solid-angle weights: +X,-X,+Z,-Z = 1.0  |  +Y,-Y (poles) = 0.552', 7.5, CX, y-10,
                 fill='#888', font_family='sans-serif', text_anchor='middle'))

# ═══ 8 — QUALITY HEADS ═══════════════════════════════════════════════
box(d, CX, y, 400, 50, C_PRED,
    ['Quality Heads  ──  [AGG] token readout'],
    ['Mean head (score) + Log-variance head (uncertainty)'])
y += 72
arrow(d, y)
arrow_label(d, y, 'train ──► loss  |  eval ──► inference')

# ═══ 9 — TRAINING LOSSES (side panel) ═════════════════════════════════
loss_y = y
lx = 20; ly = y; lw = 220; lh = 125
d.append(dw.Rectangle(lx, ly, lw, lh, fill='white', stroke=C_LOSS+'40',
                      stroke_width=1.5, rx=7, ry=7))
d.append(dw.Text('Training Losses', 9, lx+8, ly+16,
                 fill=C_LOSS, font_weight='bold', font_family='sans-serif'))
losses = [
    'L₁: Gaussian NLL (heterosced.)   w=1.0',
    'L₂: Adaptive-margin ranking      w=1.0',
    'L₃: Scene-grouped ranking        w=0.3',
    'L₄: Auxiliary face-head NLL      w=0.1',
    'Total: L₁ + L₂ + L₃ + 0.1·L₄',
]
for i, l in enumerate(losses):
    is_b = 'Total' in l
    dwg_text = dw.Text(l, 7 if not is_b else 7.5, lx+8, ly+37+i*17,
                 fill='#222' if is_b else '#444',
                 font_weight='bold' if is_b else 'normal',
                 font_family='monospace')
    d.append(dwg_text)

# Config panel
d.append(dw.Rectangle(640, ly, lw, lh, fill='white', stroke=C_TRAIN+'40',
                      stroke_width=1.5, rx=7, ry=7))
d.append(dw.Text('Training Config', 9, 648, ly+16,
                 fill=C_TRAIN, font_weight='bold', font_family='sans-serif'))
configs = [
    'Optimizer: AdamW (β₁=0.9, β₂=0.999)',
    'LR: OneCycleLR  pct_start=0.3',
    'Backbone LR: 2e-5  |  Head LR: 1e-4',
    'Batch: 2  |  Epochs: 88 max  |  Patience: 20',
    'Backbone freeze: 2 epochs',
]
for i, c in enumerate(configs):
    d.append(dw.Text(c, 7, 648, ly+37+i*17,
                     fill='#444', font_family='monospace'))

y += 10

# ═══ 10 — INFERENCE ══════════════════════════════════════════════════
inf_y = y
ih = 100; iw = 640
d.append(dw.Rectangle(gx, y, iw, ih, fill=C_INFER+'08', stroke=C_INFER+'40',
                      stroke_width=1.5, rx=10, ry=10, stroke_dasharray='6,3'))
d.append(dw.Text('Inference  (test-time)', 10, gx+12, y+18,
                 fill=C_INFER, font_weight='bold', font_family='sans-serif'))

by = y + 28
bw3 = 190
d.append(dw.Rectangle(gx+14, by, bw3, 62, fill='white', stroke=C_INFER, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('TTA Rotational Ensemble', 9, gx+14+bw3/2, by+18,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('4 yaw rotations: 0/90/180/270°', 7.5, gx+14+bw3/2, by+34,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('scores averaged → stability', 7.5, gx+14+bw3/2, by+48,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

d.append(dw.Lines(gx+14+bw3+4, by+31, gx+14+bw3+18, by+31,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+bw3+22, by, bw3, 62, fill='white', stroke=C_INFER, stroke_width=1.5, rx=6, ry=6))
d.append(dw.Text('Stereo View Fusion', 9, gx+14+bw3+22+bw3/2, by+18,
                 fill='#222', font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('L/R scores averaged', 7.5, gx+14+bw3+22+bw3/2, by+34,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('(mono: single-view bypass)', 7.5, gx+14+bw3+22+bw3/2, by+48,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

d.append(dw.Lines(gx+14+2*bw3+26, by+31, gx+14+2*bw3+40, by+31,
                  stroke='#888', stroke_width=1.5, fill='none', arrow_end=True))

d.append(dw.Rectangle(gx+14+2*bw3+44, by, 180, 62, fill=C_OUT+'15', stroke=C_OUT, stroke_width=2, rx=6, ry=6))
d.append(dw.Text('Quality Score', 11, gx+14+2*bw3+44+90, by+26,
                 fill=C_OUT, font_weight='bold', font_family='sans-serif', text_anchor='middle'))
d.append(dw.Text('per-image prediction', 7.5, gx+14+2*bw3+44+90, by+44,
                 fill='#555', font_family='sans-serif', text_anchor='middle'))

# ── Final canvas size trim ──
d.set_pixel_scale(1)

svg_path = os.path.join(OUT, 'spheriq_pipeline.svg')
png_path = os.path.join(OUT, 'spheriq_pipeline.png')
d.save_svg(svg_path)
d.save_png(png_path)
print(f"SVG: {svg_path}  ({os.path.getsize(svg_path)} bytes)")
print(f"PNG: {png_path}  ({os.path.getsize(png_path)} bytes)")
