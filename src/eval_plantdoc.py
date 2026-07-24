"""
Cross-dataset evaluation: load a ConViTX model trained on PlantVillage
(train_plantvillage.py) and test it on PlantDoc (in-field images), restricted
to the subset of classes with a clear PlantVillage equivalent
(data_prep/plantdoc_class_mapping.py) -- mirrors the paper's Table III(c).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from plantdoc_class_mapping import PLANTDOC_TO_PLANTVILLAGE

from model import ConViTX


class MappedPlantDoc(Dataset):
    def __init__(self, root_dir: str, plantvillage_classes: list, transform):
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(plantvillage_classes)}
        self.samples = []
        root = Path(root_dir)
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            pv_class = PLANTDOC_TO_PLANTVILLAGE.get(class_dir.name)
            if pv_class is None or pv_class not in self.class_to_idx:
                continue
            label = self.class_to_idx[pv_class]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((img_path, label, class_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantdoc_root", default=r"E:\plant_disease\PlantDoc_full")
    ap.add_argument("--model_dir", required=True, help="dir with best_model.pt + class_names.json from train_plantvillage.py")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split", default="test", choices=["train", "test", "both"],
                     help="PlantDoc has its own train/test split; 'both' pools them since we're not training on PlantDoc")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)

    with open(model_dir / "class_names.json") as f:
        class_names = json.load(f)

    model = ConViTX(num_classes=len(class_names), pretrained_cnn=False).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    root = Path(args.plantdoc_root)
    if args.split == "both":
        ds_train = MappedPlantDoc(root / "train", class_names, transform)
        ds_test = MappedPlantDoc(root / "test", class_names, transform)
        dataset = torch.utils.data.ConcatDataset([ds_train, ds_test])
    else:
        dataset = MappedPlantDoc(root / args.split, class_names, transform)

    print(f"PlantDoc mapped samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().numpy().tolist())
            y_true.extend(y.numpy().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    used_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    used_names = [class_names[i] for i in used_labels]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=used_labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=used_labels)
    report = classification_report(
        y_true, y_pred, labels=used_labels, target_names=used_names, digits=4, zero_division=0
    )

    print("\n=== PlantDoc cross-dataset results ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision (macro): {precision:.4f}")
    print(f"recall (macro): {recall:.4f}")
    print(f"f1 (macro): {f1:.4f}")
    print(report)

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "class_names": used_names,
        "classification_report": report,
        "n_samples": len(dataset),
    }
    out_path = model_dir / "plantdoc_crosseval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
