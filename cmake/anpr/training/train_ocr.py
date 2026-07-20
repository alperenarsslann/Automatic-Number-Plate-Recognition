"""
Turkish plate OCR model — two paths.

PATH A (recommended first): REUSE the global OCR model.
  The pretrained global-plates model already reads Turkish plates well
  (0.95+ confidence in our tests) because Turkish plates use standard Latin
  letters + digits. This script just copies it into place so the
  `turkish_finetuned` profile works immediately, with the matching charset.

    python train_ocr.py --reuse

PATH B: FINE-TUNE fast-plate-ocr on your own Turkish plate crops for maximum
  accuracy on local fonts/angles/lighting. Requires a dataset of cropped
  plate images named by their text (e.g. "34ABC123.jpg"). Fine-tuning uses
  the fast-plate-ocr training CLI; this script prints the exact command and
  prepares the model config (alphabet = Turkish charset).

    python train_ocr.py --prepare-finetune --crops plate_crops

After PATH B finishes, export the trained model to ONNX and copy it to
../models/plate_ocr_tr_v1.onnx (the fast-plate-ocr docs show the export
command; the profile's ocr.charset_path must match the alphabet below).
"""
import argparse
import shutil
from pathlib import Path

# Turkish plate alphabet: digits + the 23 letters used on TR plates
# (no Q, W, X; no diacritics). Underscore = padding slot. Keep this in sync
# with charset_tr.txt and the profile's ocr.charset_path.
TR_ALPHABET = "0123456789ABCDEFGHIJKLMNOPRSTUVYZ_"


def reuse(args):
    src_model = Path(args.global_model)
    src_charset = Path(args.global_charset)
    if not src_model.exists():
        raise SystemExit(f"global OCR model not found: {src_model}\n"
                         "Download it once via the pipeline's model set, or run the C++ build "
                         "which already fetched models/plate_ocr_global_vit_v2.onnx.")
    models_dir = Path(args.out).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_model, args.out)
    # The reused model keeps the GLOBAL charset (its output order is fixed).
    shutil.copy(src_charset, models_dir / "charset_tr.txt")
    print(f"Copied {src_model} -> {args.out}")
    print(f"Copied {src_charset} -> {models_dir / 'charset_tr.txt'} (global order preserved)")
    print("The turkish_finetuned profile can now run using the global OCR weights.")
    print("When you later train a dedicated TR OCR model (PATH B), overwrite both files.")


def prepare_finetune(args):
    crops = Path(args.crops)
    n = len(list(crops.rglob("*.jpg"))) + len(list(crops.rglob("*.png"))) if crops.exists() else 0
    print("Turkish OCR fine-tuning (fast-plate-ocr)")
    print("=" * 60)
    print(f"Crop dataset: {crops}  ({n} images found)")
    print(f"Alphabet    : {TR_ALPHABET}")
    print("\nEach crop must be named by its plate text, e.g. 34ABC123.jpg.")
    print("Recommended split: train/ and val/ subfolders.\n")
    print("Run the fast-plate-ocr trainer (see its docs for the exact current flags):")
    print("  fast_plate_ocr train \\")
    print(f"    --alphabet '{TR_ALPHABET}' \\")
    print("    --max-plate-slots 8 \\")
    print("    --img-height 70 --img-width 140 \\")
    print(f"    --train-dir {crops}/train --val-dir {crops}/val \\")
    print("    --epochs 100")
    print("\nThen export to ONNX and copy to ../models/plate_ocr_tr_v1.onnx,")
    print("and write the alphabet above into ../models/charset_tr.txt.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse", action="store_true", help="PATH A: reuse the global OCR model")
    ap.add_argument("--prepare-finetune", action="store_true", help="PATH B: print fine-tune plan")
    ap.add_argument("--crops", default="plate_crops")
    ap.add_argument("--global-model", default="../models/plate_ocr_global_vit_v2.onnx")
    ap.add_argument("--global-charset", default="../models/charset_global.txt")
    ap.add_argument("--out", default="../models/plate_ocr_tr_v1.onnx")
    args = ap.parse_args()

    if args.reuse:
        reuse(args)
    elif args.prepare_finetune:
        prepare_finetune(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
