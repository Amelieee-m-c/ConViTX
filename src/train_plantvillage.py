"""
ConViTX reproduction -- train/evaluate on the full 38-class PlantVillage dataset
(paper's Table III(a) benchmark).

Data: reuses the existing 7:2:1 split at
rethinking_fewshot_vlms/data/PlantVillage_Split_721 (read-only source, not
modified -- see repo-wide "no source edits" convention).

Protocol per paper Section III (top of page, right after Table info):
  - Input 224x224x3, rescale to [0,1] (divide by 255), no other normalization
  - Optimizer: Adam, lr=1e-4
  - Batch size: 16
  - Loss: categorical cross-entropy
Epoch count / early-stopping patience are NOT stated in the paper excerpt
available here -- we use a documented default (max 30 epochs, early stopping
patience=5 on val_loss), consistent with the "known deviations" documentation
convention used in PiTLiD_repro.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from model import ConViTX, count_params


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_transforms(img_size: int, strong_aug: bool = False):
    # paper: rescale-only ([0,1]), no ImageNet mean/std normalization
    train_ops = [transforms.Resize((img_size, img_size))]
    if strong_aug:
        # rotation/shift/zoom/shear, matching the paper's stated augmentation
        # list (not otherwise applied here) -- added for small-dataset runs
        # (e.g. PlantDoc's 1983-image train split) prone to overfitting.
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        ]
    else:
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.ToTensor())
    train_tf = transforms.Compose(train_ops)
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        y_pred.extend(out.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(cm, class_names, out_path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (ConViTX - PlantVillage)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=r"E:\plant_disease\rethinking_fewshot_vlms\data\PlantVillage_Split_721")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size_train", type=int, default=16)
    ap.add_argument("--batch_size_eval", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--strong_aug", action="store_true",
                     help="rotation/shift/zoom/shear augmentation, for small/overfit-prone splits")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_batches_per_epoch", type=int, default=0,
                     help="0 = full epoch; set >0 to cap for quick smoke tests")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_tf, eval_tf = build_transforms(args.img_size, strong_aug=args.strong_aug)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    assert train_ds.classes == val_ds.classes == test_ds.classes
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"classes: {num_classes}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_train, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_eval, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size_eval, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)

    model = ConViTX(num_classes=num_classes).to(device)
    print("trainable params:", count_params(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        if args.max_train_batches_per_epoch > 0:
            # smoke-test mode: truncate the loader
            import itertools
            limited = itertools.islice(train_loader, args.max_train_batches_per_epoch)
            model.train()
            total_loss, total_correct, total_n = 0.0, 0, 0
            for x, y in limited:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_correct += (out.argmax(1) == y).sum().item()
                total_n += x.size(0)
            train_loss, train_acc = total_loss / total_n, total_correct / total_n
        else:
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    x_axis = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x_axis, [h["train_acc"] for h in history], "o-", label="train")
    axes[0].plot(x_axis, [h["val_acc"] for h in history], "o-", label="val")
    axes[0].set_ylabel("accuracy"); axes[0].legend(); axes[0].set_title("Accuracy vs epoch")
    axes[1].plot(x_axis, [h["train_loss"] for h in history], ".-", label="train")
    axes[1].plot(x_axis, [h["val_loss"] for h in history], ".-", label="val")
    axes[1].set_ylabel("loss"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("Loss vs epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_loss.png", dpi=150)
    plt.close(fig)

    y_true, y_pred = predict_all(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    print("\n=== Test set results ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision (macro): {precision:.4f}")
    print(f"recall (macro): {recall:.4f}")
    print(f"f1 (macro): {f1:.4f}")
    print(report)

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report,
        "best_val_loss": best_val_loss,
        "stopped_epoch": len(history),
        "trainable_params": count_params(model),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
