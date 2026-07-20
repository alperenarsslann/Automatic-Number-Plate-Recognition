# Third-Party Notices

ANPR is licensed under the MIT License. It uses and/or distributes the
third-party components below, each under its own license. This file is
informational; consult each project for authoritative license text.

## Build & runtime libraries

| Component | Purpose | License |
|---|---|---|
| [OpenCV](https://opencv.org) | Video decode, DNN inference, imaging | Apache-2.0 |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON config & messages | MIT |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | Embedded REST API server | MIT |
| [FFmpeg](https://ffmpeg.org) (via OpenCV) | RTSP / H.264 / H.265 decode | LGPL-2.1+ |
| [.NET / WPF](https://github.com/dotnet/wpf) | ANPR Studio | MIT |

## Machine-learning models

The default models are downloaded by `cmake/anpr/models/prepare_models.py` and are
**not** redistributed in this repository. Each carries its own license:

| Model | Source | License |
|---|---|---|
| Plate detection (YOLOv8n) | [ml-debi/yolov8-license-plate-detection](https://huggingface.co/ml-debi/yolov8-license-plate-detection) | see model card |
| Vehicle detection (YOLOv8n, COCO) | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| Plate OCR (global MobileViT) | [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) | see project |

> **Note on Ultralytics / YOLOv8:** Ultralytics is AGPL-3.0. Models exported with
> it (including any Turkish detector you fine-tune) inherit AGPL obligations for
> that model artifact. This does not affect the MIT license of the ANPR source
> code, but review AGPL terms before commercial redistribution of the models, or
> obtain an Ultralytics commercial license.

## Training toolchain (development only)

[Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0),
[PyTorch](https://pytorch.org) (BSD-style), [ONNX](https://onnx.ai) (Apache-2.0),
[ONNX Runtime](https://onnxruntime.ai) (MIT).

## Vendor SDK (not distributed)

The Hikvision Device Network SDK (`docs/EN-HCNetSDK...`) is proprietary and is
**excluded** from this repository. ANPR connects to Hikvision cameras over
standard RTSP and does not link the SDK; it is only referenced for camera
discovery research. Obtain it from Hikvision under their license if needed.

## Datasets (not distributed)

Sample videos and the Turkish plate dataset live under `docs/dataset/` and are
git-ignored. The Turkish detection dataset is
[License Plates of Vehicles in Turkey](https://universe.roboflow.com/tr-plaka-recognition/license-plates-of-vehicles-in-turkey-s3tbj-s5lcc)
(Roboflow, CC BY 4.0).
