import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from spheriq.splits import get_scene_fn

DATA_CFG = [
    ("cviq", "cviq_scores.csv", "cviq"),
    ("live", "live_scores.csv", "live"),
    ("jufe", "jufe_scores.csv", "jufe"),
    ("odi", "odi_scores.csv", "odi"),
]

IM_EXT = {
    "cviq": ".png",
    "live": ".png",
    "jufe": ".png",
    "odi": ".jpg",
}

def load_scores(score_file: str) -> list[dict]:
    with open(score_file) as f:
        return list(csv.DictReader(f))

def build_scene_index(dataset_name: str, score_file: str) -> dict[str, list[str]]:
    scene_fn = get_scene_fn(dataset_name)
    rows = load_scores(score_file)
    index: dict[str, list[str]] = {}
    for row in rows:
        sid = scene_fn(row["image_name"])
        index.setdefault(sid, []).append(row["image_name"])
    return index

def try_load_image(base_dir: str, image_name: str, ext: str):
    path = Path(base_dir) / f"{image_name}{ext}"
    if path.exists():
        return Image.open(path).convert("RGB")
    for candidate_ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path2 = Path(base_dir) / f"{image_name}{candidate_ext}"
        if path2.exists():
            return Image.open(path2).convert("RGB")
    raise FileNotFoundError(f"Could not find image for {image_name} in {base_dir}")

def main():
    random.seed(42)
    NUM_ROWS = 4
    out_path = Path(__file__).parent / "split_verification_grid.png"

    all_datasets = []
    for ds_name, score_file, img_dir in DATA_CFG:
        scene_index = build_scene_index(ds_name, score_file)
        scenes = list(scene_index.keys())
        chosen = random.sample(scenes, min(NUM_ROWS, len(scenes)))
        samples = []
        for sid in chosen:
            img_name = random.choice(scene_index[sid])
            img = try_load_image(img_dir, img_name, IM_EXT[ds_name])
            samples.append((sid, img_name, img))
        all_datasets.append((ds_name, samples))

    cols = len(all_datasets)
    rows = max(len(s[1]) for s in all_datasets)

    THUMB_W, THUMB_H = 256, 256
    PAD = 10
    HEADER_H = 40
    LABEL_H = 30
    cell_w = THUMB_W + PAD * 2
    cell_h = THUMB_H + PAD * 2 + LABEL_H
    W = cols * cell_w
    H = HEADER_H + rows * cell_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = small_font = ImageFont.load_default()

    canvas = Image.new("RGB", (W, H), (35, 35, 35))
    draw = ImageDraw.Draw(canvas)

    for ci, (ds_name, samples) in enumerate(all_datasets):
        cx = ci * cell_w
        draw.text((cx + PAD, 8), ds_name.upper(), font=font, fill=(200, 200, 100))

        for ri, (sid, img_name, img) in enumerate(samples):
            cy = HEADER_H + ri * cell_h

            thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            canvas.paste(thumb, (cx + PAD, cy + PAD))

            label = f"scene={sid}  ({img_name[:25]})"
            draw.text((cx + PAD, cy + THUMB_H + PAD + 2), label, font=small_font, fill=(220, 220, 220))

    canvas.save(out_path, quality=95)
    print(f"Saved verification grid → {out_path}")
    print(f"Grid: {cols} datasets × {rows} scenes")
    for ds_name, samples in all_datasets:
        scenes = [s[0] for s in samples]
        print(f"  {ds_name}: scenes = {scenes}")

if __name__ == "__main__":
    main()
