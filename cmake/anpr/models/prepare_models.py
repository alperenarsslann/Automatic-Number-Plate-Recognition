"""
Download / prepare the default ONNX models the ANPR pipeline needs.

The models are not committed to git (they are large binaries); run this once
after cloning to populate this folder. Idempotent — existing files are skipped.

    python prepare_models.py            # download plate detection + OCR
    python prepare_models.py --vehicle  # also export the COCO vehicle model
                                        # (requires: pip install ultralytics)

Produces (matching config/anpr.example.json):
    plate_detect_yolov8n.onnx        plate detection
    plate_ocr_global_vit_v2.onnx     OCR (global, reads Turkish plates too)
    vehicle_detect_yolov8n_416.onnx  vehicle detection (with --vehicle)

Charset files (charset_global.txt, charset_tr.txt) are tracked in git already.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> download URL
DOWNLOADS = {
    "plate_detect_yolov8n.onnx":
        "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx",
    "plate_ocr_global_vit_v2.onnx":
        "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/global_mobile_vit_v2_ocr.onnx",
}


def _progress(block_num, block_size, total):
    if total > 0:
        pct = min(100, block_num * block_size * 100 // total)
        sys.stdout.write(f"\r    {pct:3d}%")
        sys.stdout.flush()


def download(name: str, url: str) -> None:
    dest = HERE / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name} already present")
        return
    print(f"[get ] {name}")
    urllib.request.urlretrieve(url, dest, _progress)
    print(f"\r[ok  ] {name} ({dest.stat().st_size // 1024} KB)")


def export_vehicle() -> None:
    dest = HERE / "vehicle_detect_yolov8n_416.onnx"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present")
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[warn] ultralytics not installed — cannot export the vehicle model.")
        print("       Either `pip install ultralytics` and re-run with --vehicle,")
        print("       or set vehicle_detection.enabled=false in your config.")
        return
    print("[get ] exporting COCO yolov8n -> vehicle_detect_yolov8n_416.onnx")
    exported = YOLO("yolov8n.pt").export(format="onnx", opset=12, imgsz=416, dynamic=False)
    Path(exported).replace(dest)
    # ultralytics also drops yolov8n.pt in cwd; leave it for the user to clean.
    print(f"[ok  ] {dest.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", action="store_true",
                    help="also export the COCO vehicle detection model (needs ultralytics)")
    args = ap.parse_args()

    for name, url in DOWNLOADS.items():
        try:
            download(name, url)
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"\n[fail] {name}: {e}")

    if args.vehicle:
        export_vehicle()
    else:
        print("\nTip: run with --vehicle to also produce the vehicle detection model,")
        print("     or set vehicle_detection.enabled=false in your config.")

    print("\nDone. Models are in:", HERE)


if __name__ == "__main__":
    main()
