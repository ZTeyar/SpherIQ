import hashlib
import csv
import random
from typing import Callable


def get_stable_seed(input_string: str) -> int:
    hash_object = hashlib.md5(input_string.encode('utf-8'))
    return int(hash_object.hexdigest(), 16) % (2 ** 32)


SCENE_FN_REGISTRY: dict[str, Callable[[str], str]] = {
    'live':      lambda name: name.split('_')[0],
    'live_3d':   lambda name: name.split('_')[0],
    'jufe':      lambda name: name.split('_')[0],
    'cviq':      lambda name: str(((int(name.replace('.png', '').replace('.jpg', '')) - 1) // 34) + 1),
    'odi':       lambda name: '_'.join(name.split('_')[-2:]),
    'oiqa':      lambda name: name.split('_')[0],
    'ivqad':     lambda name: name.split('_')[0],
}


def get_scene_fn(dataset_name: str) -> Callable[[str], str]:
    lower = dataset_name.lower()
    for key, fn in SCENE_FN_REGISTRY.items():
        if key in lower:
            return fn
    raise KeyError(
        f"No scene_fn registered for dataset '{dataset_name}'. "
        f"Add it to SCENE_FN_REGISTRY in splits.py"
    )


def get_scene_ids(dataset_name: str, score_file: str) -> list[str]:
    scene_fn = get_scene_fn(dataset_name)
    scenes = []
    with open(score_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenes.append(scene_fn(row['image_name']))
    return scenes


def get_fold_split(
    dataset_name: str,
    score_file: str,
    num_folds: int = 5,
    fold_index: int = 0,
) -> tuple[set[str], set[str]]:
    scenes = get_scene_ids(dataset_name, score_file)
    unique_scenes = sorted(set(scenes))
    rng = random.Random(get_stable_seed(dataset_name))
    rng.shuffle(unique_scenes)

    fold_size = len(unique_scenes) // num_folds
    val_start = fold_index * fold_size
    val_end = val_start + fold_size if fold_index < num_folds - 1 else len(unique_scenes)
    val_scenes = set(unique_scenes[val_start:val_end])
    train_scenes = set(unique_scenes) - val_scenes
    return train_scenes, val_scenes


def get_standard_split(
    dataset_name: str,
    score_file: str,
    train_ratio: float = 0.8,
) -> tuple[set[str], set[str]]:
    scenes = get_scene_ids(dataset_name, score_file)
    unique_scenes = sorted(set(scenes))
    rng = random.Random(get_stable_seed(dataset_name))
    rng.shuffle(unique_scenes)

    split_idx = int(train_ratio * len(unique_scenes))
    train_scenes = set(unique_scenes[:split_idx])
    val_scenes = set(unique_scenes[split_idx:])
    return train_scenes, val_scenes
