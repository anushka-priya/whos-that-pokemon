import time

import torch
from torch.utils.data import DataLoader

from . import config
from .losses import DiceBCELoss, dice_iou_metrics
from .models import UNet


def train_segmentation_model(train_ds, val_ds, device):
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNet(base=32).to(device)
    loss_fn = DiceBCELoss()

    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    best_val_dice = -1.0

    for epoch in range(config.EPOCHS):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_dice_sum, val_iou_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                d, i = dice_iou_metrics(logits, masks)
                val_dice_sum += d * imgs.size(0)
                val_iou_sum += i * imgs.size(0)
                n += imgs.size(0)

        val_dice = val_dice_sum / n
        val_iou = val_iou_sum / n

        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), config.SEG_CHECKPOINT)

        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | time={time.time()-t0:.1f}s")

    print(f"\nBest validation Dice: {best_val_dice:.4f} (checkpoint saved to {config.SEG_CHECKPOINT})")
    return model, history