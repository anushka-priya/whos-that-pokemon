import os
import csv
import json
import urllib.request

import numpy as np
from PIL import Image

from . import config


def download_sprites(start_id=1, end_id=151):
    for pid in range(start_id, end_id + 1):
        out_path = f"{config.RAW_DIR}/{pid}.png"
        if os.path.exists(out_path):
            continue
        url = f"{config.SPRITE_BASE_URL}/{pid}.png"
        urllib.request.urlretrieve(url, out_path)
    print(f"Downloaded {len(os.listdir(config.RAW_DIR))} sprites")


def load_names():
    urllib.request.urlretrieve(config.NAMES_CSV_URL, "pokemon_names_raw.csv")
    id_to_name = {}
    with open("pokemon_names_raw.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["id"])
            if 1 <= pid <= 151:
                id_to_name[pid] = row["identifier"]
    print(f"Loaded {len(id_to_name)} names")
    return id_to_name


def random_solid_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())


def composite_on_solid_bg(sprite_path, bg_color, out_size=config.OUT_SIZE):
    img = Image.open(sprite_path).convert("RGBA").resize((out_size, out_size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4] / 255.0

    bg = np.ones((out_size, out_size, 3), dtype=np.float32) * np.array(bg_color, dtype=np.float32)
    composited = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    mask = (arr[:, :, 3] > 127).astype(np.uint8) * 255

    return Image.fromarray(composited), Image.fromarray(mask)


def build_dataset(id_to_name):
    manifest = []
    for pid in range(1, 152):
        for i in range(config.N_BG_PER_POKEMON):
            bg = random_solid_color()
            comp, mask = composite_on_solid_bg(f"{config.RAW_DIR}/{pid}.png", bg)
            fname = f"{pid}_{i}.png"
            comp.save(f"{config.IMG_DIR}/{fname}")
            mask.save(f"{config.MASK_DIR}/{fname}")
            manifest.append({"filename": fname, "pokemon_id": pid, "name": id_to_name[pid]})

    with open("dataset/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} image/mask pairs across {len(id_to_name)} Pokemon")
    return manifest