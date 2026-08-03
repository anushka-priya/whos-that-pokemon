import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from . import config


def show_sample_sprite(pokemon_id=25, name="Pikachu"):
    sample = Image.open(f"{config.RAW_DIR}/{pokemon_id}.png")
    print("Mode:", sample.mode, "| Size:", sample.size)
    assert sample.mode == "RGBA", "Expected an alpha channel — check the download URL/path"

    plt.figure(figsize=(3, 3))
    plt.imshow(sample)
    plt.title(name)
    plt.axis("off")
    plt.show()


def show_composite_example(manifest, pokemon_id=25):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    entry = [m for m in manifest if m["pokemon_id"] == pokemon_id][0]
    img = Image.open(f"{config.IMG_DIR}/{entry['filename']}")
    mask = Image.open(f"{config.MASK_DIR}/{entry['filename']}")

    axes[0].imshow(Image.open(f"{config.RAW_DIR}/{pokemon_id}.png"))
    axes[0].set_title("Original")
    axes[1].imshow(img)
    axes[1].set_title("Composited")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground truth mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_loss_curves(history, kind="segmentation"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"{kind.title()} Train Loss")
    axes[0].set_xlabel("Epoch")

    if kind == "segmentation":
        axes[1].plot(history["val_dice"], label="Val Dice")
        axes[1].plot(history["val_iou"], label="Val IoU")
        axes[1].set_title("Validation Metrics")
    else:
        axes[1].plot(history["train_acc"], label="Train acc")
        axes[1].plot(history["val_acc"], label="Val acc")
        axes[1].set_title("Classifier Accuracy")

    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def show_seg_predictions(model, val_ds, device, n_examples=5):
    import torch
    fig, axes = plt.subplots(n_examples, 3, figsize=(7, 3 * n_examples))
    sample_indices = random.sample(range(len(val_ds)), n_examples)

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            img, mask = val_ds[idx]
            entry = val_ds.manifest[idx]

            logits = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()

            axes[row, 0].imshow(img.permute(1, 2, 0).numpy())
            axes[row, 0].set_title(f"Input ({entry['name']})")
            axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray")
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].imshow(pred, cmap="gray")
            axes[row, 2].set_title("Prediction")

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.show()
    return sample_indices


def show_silhouette_reveal(model, val_ds, device, idx):
    import torch
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    img, _ = val_ds[idx]
    entry = val_ds.manifest[idx]

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logits) > 0.5).cpu().squeeze().numpy()

    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    silhouette = np.zeros_like(img_np)
    silhouette[pred] = [0, 0, 0]
    silhouette[~pred] = [255, 255, 255]

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Predicted mask")
    axes[2].imshow(silhouette)
    axes[2].set_title(f"Who's that Pokemon?\n(answer: {entry['name']})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_demo_grid(predict_fn, val_ds, n_examples=4):
    fig, axes = plt.subplots(n_examples, 3, figsize=(8, 10))
    demo_indices = random.sample(range(len(val_ds)), n_examples)

    for row, idx in enumerate(demo_indices):
        img, _ = val_ds[idx]
        entry = val_ds.manifest[idx]
        true_name = entry["name"]

        pred_name, pred_mask = predict_fn(img)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("input")
        axes[row, 1].imshow(pred_mask, cmap="gray")
        axes[row, 1].set_title("Predicted silhouette")

        correct = (pred_name == true_name)
        color = "green" if correct else "red"
        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 2].set_title(f"Guess: {pred_name}\nActual: {true_name}", color=color, fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()