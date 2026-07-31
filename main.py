import random

import numpy as np
import torch

from whos_that_pokemon import config
from whos_that_pokemon.data_prep import download_sprites, load_names, build_dataset
from whos_that_pokemon.datasets import PokemonSegDataset, SilhouetteDataset
from whos_that_pokemon.evaluate import make_seg_splits, make_cls_splits, evaluate_end_to_end
from whos_that_pokemon.train_segmentation import train_segmentation_model
from whos_that_pokemon.train_classifier import train_classifier_model
from whos_that_pokemon.pipeline import load_pipeline, make_predict_fn
from whos_that_pokemon.visualize import plot_loss_curves, show_seg_predictions, show_demo_grid


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config.ensure_dirs()

    #data
    download_sprites(1, 151)
    id_to_name = load_names()
    manifest = build_dataset(id_to_name)
    train_manifest, val_manifest = make_seg_splits(manifest)

    #segmentation
    train_ds = PokemonSegDataset("dataset/train_manifest.json")
    val_ds = PokemonSegDataset("dataset/val_manifest.json")
    seg_model, seg_history = train_segmentation_model(train_ds, val_ds, device)
    plot_loss_curves(seg_history, kind="segmentation")
    show_seg_predictions(seg_model, val_ds, device)

    #classification
    cls_train_manifest, cls_val_manifest, label_maps = make_cls_splits(manifest)
    cls_train_ds = SilhouetteDataset(cls_train_manifest, label_maps["id_to_label"])
    cls_val_ds = SilhouetteDataset(cls_val_manifest, label_maps["id_to_label"])
    classifier, cls_history = train_classifier_model(
        cls_train_ds, cls_val_ds, label_maps["num_classes"], device
    )
    plot_loss_curves(cls_history, kind="classifier")

    #pipeline
    seg_model, classifier = load_pipeline(label_maps["num_classes"], device)
    predict_fn = make_predict_fn(seg_model, classifier, label_maps["label_to_name"], device)
    evaluate_end_to_end(predict_fn, val_ds)
    show_demo_grid(predict_fn, val_ds)


if __name__ == "__main__":
    main()