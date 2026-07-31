import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .models import PokemonClassifier


def train_classifier_model(train_ds, val_ds, num_classes, device):
    train_loader = DataLoader(train_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=2)

    classifier = PokemonClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.CLS_LR)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(config.CLS_EPOCHS):
        classifier.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for masks, labels in train_loader:
            masks, labels = masks.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = classifier(masks)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * masks.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += masks.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for masks, labels in val_loader:
                masks, labels = masks.to(device), labels.to(device)
                logits = classifier(masks)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += masks.size(0)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), config.CLS_CHECKPOINT)

        print(f"Epoch {epoch+1:3d}/{config.CLS_EPOCHS} | train_loss={train_loss:.4f} | "
              f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | time={time.time()-t0:.1f}s")

    print(f"\nBest validation accuracy: {best_val_acc:.3f} (checkpoint saved to {config.CLS_CHECKPOINT})")
    return classifier, history