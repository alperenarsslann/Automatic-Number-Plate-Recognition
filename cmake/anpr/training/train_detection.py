"""
Fine-tune a Turkish plate DETECTION model (YOLOv8) and export it as the
ONNX file the C++ pipeline's `turkish_finetuned` profile expects.

Starts from the pretrained yolov8n plate detector (transfer learning), so a
few hundred well-labeled Turkish images already give a strong model.

Example (GPU):
  python train_detection.py --data data.yaml --epochs 100 --imgsz 640 --device 0
Output:
  ../models/plate_detect_tr_v1.onnx        (drop-in for the profile)
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml", help="YOLO dataset config")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="starting weights (yolov8n.pt, or a plate-pretrained .pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="must match the profile input size")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' otherwise")
    ap.add_argument("--out", default="../models/plate_detect_tr_v1.onnx")
    ap.add_argument("--name", default="tr_plate_detect")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, name=args.name)

    # Export best weights to ONNX with a fixed input size (opset 12 loads
    # cleanly in OpenCV's DNN module, matching the runtime).
    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    exported = YOLO(str(best)).export(format="onnx", opset=12, imgsz=args.imgsz, dynamic=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    print(f"\nDone. Copied {exported} -> {out}")
    print("Set processing.active_model_profile to 'turkish_finetuned' and check that the "
          "profile's detection.input_width/height match --imgsz.")


if __name__ == "__main__":
    main()
