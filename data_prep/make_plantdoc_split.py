"""
Builds a train/val/test split for PlantDoc's own 27-class classification task
(paper's Table III(c) protocol: trained AND tested on PlantDoc itself, not a
zero-shot transfer from a PlantVillage-trained model).

PlantDoc_full/ (from the official pratikkayal/PlantDoc-Dataset repo) ships its
own train/test folders but no val split. This carves a stratified val split
out of train (default 15%) and copies test through unchanged. The one class
present only in train ("Tomato two spotted spider mites leaf", no test
images) is dropped so train/val/test share an identical 27-class label set.
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path


def safe_copy(src_file: Path, dst_dir: Path):
    """Copy, proactively shortening the filename if the full destination path
    would be near Windows' legacy MAX_PATH (260 chars) -- PlantDoc has several
    very long descriptive filenames, and while directory listing (iterdir)
    tolerates near-260-char paths, plain open() calls (as used by PIL /
    DataLoader workers) can still fail on them."""
    dst = dst_dir / src_file.name
    if len(str(dst)) > 200:
        short_name = hashlib.md5(src_file.name.encode()).hexdigest()[:16] + src_file.suffix
        dst = dst_dir / short_name
    shutil.copy2(src_file, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default=r"E:\plant_disease\PlantDoc_full")
    ap.add_argument("--output_dir", default=r"E:\plant_disease\ConViTX_repro\data\plantdoc_split")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.source_dir)
    out = Path(args.output_dir)

    train_classes = sorted(p.name for p in (src / "train").iterdir() if p.is_dir())
    test_classes = set(p.name for p in (src / "test").iterdir() if p.is_dir())
    classes = sorted(c for c in train_classes if c in test_classes)
    dropped = sorted(set(train_classes) - set(classes))
    print(f"classes used: {len(classes)} (dropped, no test images: {dropped})")

    for split in ["train", "val", "test"]:
        for c in classes:
            (out / split / c).mkdir(parents=True, exist_ok=True)

    report = []
    for c in classes:
        imgs = [f for f in (src / "train" / c).iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        random.shuffle(imgs)
        n_val = max(1, round(len(imgs) * args.val_frac))
        val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

        for f in train_imgs:
            safe_copy(f, out / "train" / c)
        for f in val_imgs:
            safe_copy(f, out / "val" / c)

        test_imgs = [f for f in (src / "test" / c).iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        for f in test_imgs:
            safe_copy(f, out / "test" / c)

        report.append((c, len(train_imgs), len(val_imgs), len(test_imgs)))

    print(f"{'class':55s} {'train':>6} {'val':>5} {'test':>5}")
    for c, tr, va, te in report:
        print(f"{c:55s} {tr:6d} {va:5d} {te:5d}")
    print(f"\nTOTAL train={sum(r[1] for r in report)} val={sum(r[2] for r in report)} test={sum(r[3] for r in report)}")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
