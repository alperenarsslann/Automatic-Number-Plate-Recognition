"""
Prepare a Turkish plate DETECTION dataset in YOLO format.

Two modes:

  --bootstrap:  Use the existing generic plate detector to PRE-LABEL your raw
                Turkish images (weak labels). You then correct the boxes in a
                tool like labelImg / Roboflow / CVAT. This turns "label 2000
                images from scratch" into "fix a few boxes each".

  --split:      Split an already-labeled image+label set into train/val.

Example:
  # 1) drop raw images into training/raw_images/, then:
  python prepare_dataset.py --bootstrap --images raw_images --out dataset \
         --detector ../models/plate_detect_yolov8n.onnx
  # 2) review/fix labels in labelImg (YOLO format), then:
  python prepare_dataset.py --split --dataset dataset --val-fraction 0.15
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return canvas, scale, px, py


def detect_plates(net, img, conf=0.30, size=640):
    """Run the generic ONNX plate detector, return boxes as (x, y, w, h) in
    source pixels. Handles YOLOv8 output layout [1, 4+nc, N]."""
    boxed, scale, px, py = letterbox(img, size)
    blob = cv2.dnn.blobFromImage(boxed, 1 / 255.0, (size, size), swapRB=True)
    net.setInput(blob)
    out = net.forward()
    pred = out[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # -> [N, 4+nc]
    boxes, scores = [], []
    for row in pred:
        score = float(row[4:].max())
        if score < conf:
            continue
        cx, cy, bw, bh = row[:4]
        x = (cx - bw / 2 - px) / scale
        y = (cy - bh / 2 - py) / scale
        boxes.append([x, y, bw / scale, bh / scale])
        scores.append(score)
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45)
    return [boxes[i] for i in np.array(idx).flatten()] if len(idx) else []


def bootstrap(args):
    detector = Path(args.detector)
    if not detector.exists():
        raise SystemExit(f"detector not found: {detector}")
    net = cv2.dnn.readNetFromONNX(str(detector))

    out_root = Path(args.out)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    images = [p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"Bootstrapping weak labels for {len(images)} image(s)...")
    labeled = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = detect_plates(net, img, conf=args.conf)
        shutil.copy(path, out_root / "images" / path.name)
        label_path = (out_root / "labels" / path.name).with_suffix(".txt")
        with open(label_path, "w") as f:
            for (x, y, bw, bh) in boxes:
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}\n")
        labeled += 1
    print(f"Wrote {labeled} images + weak labels to {out_root}/")
    print("NEXT: open them in labelImg (YOLO mode) and CORRECT the boxes, then run --split.")


def split(args):
    root = Path(args.dataset)
    images = sorted(p for p in (root / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)
    random.seed(args.seed)
    random.shuffle(images)
    n_val = int(len(images) * args.val_fraction)
    for sub in ("train", "val"):
        (root / "images" / sub).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sub).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        sub = "val" if i < n_val else "train"
        label = (root / "labels" / img.name).with_suffix(".txt")
        img.rename(root / "images" / sub / img.name)
        if label.exists():
            label.rename(root / "labels" / sub / label.name)
    print(f"Split {len(images)} images -> {len(images) - n_val} train / {n_val} val")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", action="store_true", help="pre-label raw images")
    ap.add_argument("--split", action="store_true", help="split into train/val")
    ap.add_argument("--images", default="raw_images", help="raw image folder (bootstrap)")
    ap.add_argument("--out", default="dataset", help="output dataset folder (bootstrap)")
    ap.add_argument("--dataset", default="dataset", help="dataset folder (split)")
    ap.add_argument("--detector", default="../models/plate_detect_yolov8n.onnx")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.bootstrap:
        bootstrap(args)
    elif args.split:
        split(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
