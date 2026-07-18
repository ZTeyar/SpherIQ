import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from spheriq.splits import get_scene_fn
import csv

REPO = Path(__file__).resolve().parent.parent
CVIQ_DIR = REPO / "cviq"
SCORE_FILE = REPO / "CVIQ.csv"
OUT_PATH = Path(__file__).parent / "cviq_first_variant_grid.png"

scene_fn = get_scene_fn("cviq")

first_variant: dict[str, str] = {}
with open(SCORE_FILE) as f:
    for row in csv.DictReader(f):
        name = row["Image Name"]
        sid = scene_fn(name)
        if sid not in first_variant:
            first_variant[sid] = name

sorted_scenes = sorted(first_variant.keys(), key=int)
samples = [(sid, first_variant[sid]) for sid in sorted_scenes]

NUM_COLS = 8
THUMB_W, THUMB_H = 320, 320
PAD = 8
LABEL_H = 30
HEADER_H = 40

cell_w = THUMB_W + PAD * 2
cell_h = THUMB_H + PAD * 2 + LABEL_H
NUM_ROWS = (len(samples) + NUM_COLS - 1) // NUM_COLS
W = NUM_COLS * cell_w
H = HEADER_H + NUM_ROWS * cell_h

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except Exception:
    font = small_font = ImageFont.load_default()

canvas = Image.new("RGB", (W, H), (30, 30, 30))
draw = ImageDraw.Draw(canvas)
draw.text((PAD, 8), "CVIQ — First variant of each scene (via splits.py)", font=font, fill=(200, 200, 100))

for i, (sid, img_name) in enumerate(samples):
    img_path = CVIQ_DIR / img_name
    col = i % NUM_COLS
    row = i // NUM_COLS
    cx = col * cell_w
    cy = HEADER_H + row * cell_h

    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
        thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        canvas.paste(thumb, (cx + PAD, cy + PAD))
    else:
        draw.rectangle(
            [cx + PAD, cy + PAD, cx + PAD + THUMB_W, cy + PAD + THUMB_H],
            fill=(60, 60, 60),
        )

    label = f"Scene {sid}  ({img_name})"
    draw.text((cx + PAD, cy + THUMB_H + PAD + 4), label, font=small_font, fill=(220, 220, 220))

canvas.save(OUT_PATH, quality=95)
print(f"Saved → {OUT_PATH}")
for sid, name in samples:
    print(f"  Scene {sid:>2} : {name}")
