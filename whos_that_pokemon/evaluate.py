import json
import random

from . import config

def make_seg_splits(manifest, val_fraction=config.VAL_FRACTION):
    pokemon_ids = sorted(set(m["pokemon_id"] for m in manifest))
    random.shuffle(pokemon_ids)

    n_val = int(len(pokemon_ids) * val_fraction)
    val_ids = set(pokemon_ids[:n_val])
    train_ids = set(pokemon_ids[n_val:])

    train_manifest = [m for m in manifest if m["pokemon_id"] in train_ids]
    val_manifest = [m for m in manifest if m["pokemon_id"] in val_ids]

    assert len(train_ids & val_ids) == 0, "Leakage detected: a Pokemon ID appears in both splits"

    print(f"Train: {len(train_ids)} Pokemon, {len(train_manifest)} images")
    print(f"Val:   {len(val_ids)} Pokemon, {len(val_manifest)} images")

def make_cls_splits(manifest, val_fraction=config.VAL_FRACTION):
    pokemon_ids_sorted = sorted(set(m["pokemon_id"] for m in manifest))
    id_to_label = {pid: i for i, pid in enumerate(pokemon_ids_sorted)}
    label_to_name = {
        lbl: next(m["name"] for m in manifest if m["pokemon_id"] == pid)
        for pid, lbl in id_to_label.items()
    }
    num_classes = len(pokemon_ids_sorted)
    print(f"Number of classes: {num_classes}")

    indices = list(range(len(manifest)))
    random.shuffle(indices)
    n_val = int(len(manifest) * val_fraction)

    val_manifest = [manifest[i] for i in indices[:n_val]]
    train_manifest = [manifest[i] for i in indices[n_val:]]

    print(f"Stage 2 train: {len(train_manifest)} images | val: {len(val_manifest)} images")

    label_maps = {"id_to_label": id_to_label, "label_to_name": label_to_name, "num_classes": num_classes}
    return train_manifest, val_manifest, label_maps


def evaluate_end_to_end(predict_fn, val_ds):
    correct, total = 0, 0
    for i in range(len(val_ds)):
        img, _ = val_ds[i]
        entry = val_ds.manifest[i]
        true_name = entry["name"]

        pred_name, _ = predict_fn(img)
        correct += int(pred_name == true_name)
        total += 1

    acc = correct / total
    print(f"End-to-end accuracy on unseen Pokemon: {acc:.3f} ({correct}/{total})")
    return acc