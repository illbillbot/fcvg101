"""FCVG101 Task 2: train FruitCNN with class-weighted cross-entropy.

Run from the Task 2 project root:
    python train_balanced.py --epochs 15 --batch-size 32 --lr 0.001

This keeps all training images while compensating for class imbalance. The
validation split is used to select the checkpoint; the test split is not used
for model selection.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score

from src.model import FruitCNN
from src.data_utils import IMAGE_SIZE, MEAN, STD, SEED, make_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(dataset: ImageFolder, n_classes: int, device: torch.device):
    """Return inverse-frequency weights and class counts in ImageFolder order."""
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=n_classes).to(torch.float32)
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"Training split has no samples for class indices: {missing}")
    weights = counts.sum() / (n_classes * counts)
    return weights.to(device), counts.to(torch.long)


def run_epoch(model, loader, criterion, optimizer, device, labels):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total += batch_size
            all_targets.extend(targets.detach().cpu().tolist())
            all_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())

    if total == 0:
        raise RuntimeError("DataLoader is empty. Check the dataset splits and image files.")

    mean_loss = total_loss / total
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(
        all_targets, all_predictions, labels=labels, average="macro", zero_division=0
    )
    return float(mean_loss), float(accuracy), float(macro_f1)


def main() -> None:
    parser = argparse.ArgumentParser(description="FCVG101 Task 2: class-balanced CNN training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs and batch-size must be positive; lr must be greater than zero")

    set_seed(args.seed)
    root = Path("data/splits")
    train_set = ImageFolder(root / "train", transform=make_transform(train=True))
    val_set = ImageFolder(root / "val", transform=make_transform(train=False))
    test_set = ImageFolder(root / "test", transform=make_transform(train=False))

    if train_set.class_to_idx != val_set.class_to_idx or train_set.class_to_idx != test_set.class_to_idx:
        raise RuntimeError("Class mapping differs between train, validation and test splits.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights, class_counts = calculate_class_weights(train_set, len(train_set.classes), device)
    labels = list(range(len(train_set.classes)))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FruitCNN(n_classes=len(train_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    artifacts = Path("artifacts")
    results = Path("results")
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (artifacts / "classes.json").write_text(
        json.dumps(train_set.class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    class_summary = {
        name: {
            "train_images": int(class_counts[index].item()),
            "class_weight": float(class_weights[index].item()),
        }
        for name, index in train_set.class_to_idx.items()
    }
    (results / "training_class_balance.json").write_text(
        json.dumps(class_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Classes: {train_set.classes}")
    print(f"Images train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print("Training class counts and weights:")
    for class_name, values in class_summary.items():
        print(
            f"  {class_name}: {values['train_images']} images, "
            f"weight={values['class_weight']:.4f}"
        )

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, labels
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, None, device, labels
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss_weighted": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss_weighted": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        # Prefer higher validation macro-F1; use lower weighted validation loss to break ties.
        is_better = (val_f1 > best_val_f1) or (
            np.isclose(val_f1, best_val_f1) and val_loss < best_val_loss
        )
        if is_better:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_set.class_to_idx,
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "architecture": "FruitCNN_3Conv_GAP",
                    "best_epoch": epoch,
                    "best_val_loss": val_loss,
                    "best_val_accuracy": val_acc,
                    "best_val_macro_f1": val_f1,
                    "class_counts": class_summary,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "selection_metric": "validation_macro_f1",
                    "seed": args.seed,
                },
                artifacts / "model.pth",
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_macroF1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f}"
        )

    history_path = results / "train_history.csv"
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    chart_specs = [
        ("train_loss_weighted", "val_loss_weighted", "Weighted cross-entropy loss"),
        ("train_accuracy", "val_accuracy", "Accuracy"),
        ("train_macro_f1", "val_macro_f1", "Macro F1-score"),
    ]
    epochs = [row["epoch"] for row in history]
    for ax, (train_key, val_key, title) in zip(axes, chart_specs):
        ax.plot(epochs, [row[train_key] for row in history], label="Train")
        ax.plot(epochs, [row[val_key] for row in history], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(results / "learning_curves.png", dpi=160)
    plt.close(fig)

    print(f"Best validation macro-F1: {best_val_f1:.4f} (weighted val loss={best_val_loss:.4f})")
    print("Saved: artifacts/model.pth, artifacts/classes.json")
    print("Saved: results/train_history.csv, results/learning_curves.png")
    print("Next: run python evaluate.py once model selection is complete.")


if __name__ == "__main__":
    main()
