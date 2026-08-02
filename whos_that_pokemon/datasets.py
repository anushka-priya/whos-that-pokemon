import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config


class PokemonSegDataset(Dataset):
    def __init__(self, manifest_path, img_dir=config.IMG_DIR, mask_dir=config.MASK_DIR):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        fname = entry["filename"]

        img = Image.open(f"{self.img_dir}/{fname}").convert("RGB")
        mask = Image.open(f"{self.mask_dir}/{fname}").convert("L")

        img_arr = np.array(img).astype(np.float32) / 255.0
        mask_arr = np.array(mask).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return img_tensor, mask_tensor


class SilhouetteDataset(Dataset):
    def __init__(self, manifest_subset, id_to_label, mask_dir=config.MASK_DIR):
        self.manifest = manifest_subset
        self.id_to_label = id_to_label
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        mask = Image.open(f"{self.mask_dir}/{entry['filename']}").convert("L")
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        label = self.id_to_label[entry["pokemon_id"]]
        return mask_tensor, label