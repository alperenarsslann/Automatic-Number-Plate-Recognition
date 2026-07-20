# Türk Plakası Fine-Tune Rehberi

Bu klasör, ANPR pipeline'ının `turkish_finetuned` model profilini besleyen
**Türk plakasına özel** tespit ve OCR modellerini üretmek içindir. Çıktı iki
dosyadır (config'in beklediği yerler):

- `../models/plate_detect_tr_v1.onnx` — Türk plakası tespit modeli
- `../models/plate_ocr_tr_v1.onnx` — OCR modeli (+ `../models/charset_tr.txt`)

Bu dosyalar hazır olunca C++ tarafında **hiçbir kod değişmeden** sadece
config'te `active_model_profile: "turkish_finetuned"` yapıp yeniden başlatmak
yeterlidir.

## Neden iki ayrı model?

Pipeline üç aşamalı: **araç tespiti → plaka tespiti → OCR**. Araç tespiti
generic COCO modeliyle kalır (marka/ülke fark etmez). Fine-tune ettiğimiz
kısım **plaka tespiti** (Türk plakasının görünümü, montaj açısı, oranı) ve
istenirse **OCR** (yerel font/ışık koşulları). Türk plakaları standart Latin
harf + rakam kullandığı için (Ç/Ğ/İ/Ö/Ş/Ü ve Q/W/X yok) mevcut global OCR
modeli zaten iyi çalışıyor — bu yüzden OCR için önce "yeniden kullan" yolu
öneriliyor.

## Kurulum

> **DİKKAT — Python sürümü:** PyTorch'un CUDA (GPU) wheel'leri **Python 3.14
> için henüz yok** (sadece CPU wheel'i var). GPU eğitimi için Python **3.12**
> kullanın. Sistemde yoksa: `winget install Python.Python.3.12` (mevcut
> 3.14 kurulumunuza dokunmaz, yan yana durur).

```powershell
cd training
py -3.12 -m venv .venv312
.venv312\Scripts\activate
# GPU eğitimi için CUDA'lı torch (RTX 3050 için cu121; cu124 de olur):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

GPU'nun görüldüğünü doğrulayın (çıktı `True ... RTX 3050` olmalı):
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Alternatif — Google Colab (ücretsiz T4 16GB GPU):** yerel kurulum derdi
yok ve 4GB'lik 3050'den geniş. `!pip install ultralytics` → dataset'i yükle
→ `YOLO('yolov8n.pt').train(data=..., epochs=100, imgsz=640, batch=16, device=0)`
→ `.export(format='onnx', opset=12, imgsz=640)` → inen `best.onnx`'i
`models/plate_detect_tr_v1.onnx` yap.

### Bu depoya özel Türk dataseti (hazır)

Roboflow "License Plates of Vehicles in Turkey" v1 seti
`docs/dataset/archive/License Plates of Vehicles in Turkey.v1i.yolov8/`
altında (3150 train / 345 val, YOLOv8 formatı). `training/data_tr.yaml` buna
işaret eder — **Adım 1 hazırlığı gerekmez**, doğrudan Adım 2'ye geçin:
```powershell
python train_detection.py --data data_tr.yaml --epochs 100 --imgsz 640 --batch 8 --device 0
```
(`--batch 8`: RTX 3050 4GB için; Colab T4'te 16 kullanın.)

## Adım 1 — Veri seti

Türk plakası tespit veri setine ihtiyacınız var. Kaynaklar:
- Kendi kameranızdan topladığınız kareler (en iyisi — gerçek dağılım),
- Açık veri setleri (Roboflow "turkish license plate", Kaggle TR plaka setleri).

Ham görüntüleri `training/raw_images/` altına koyun. Etiketleme yükünü
azaltmak için **zayıf etiket bootstrap**'ı kullanın — mevcut generic dedektör
görüntüleri ön-etiketler, siz sadece düzeltirsiniz:

```powershell
python prepare_dataset.py --bootstrap --images raw_images --out dataset
```

Sonra `dataset/labels/` altındaki kutuları **labelImg** (YOLO modu) / Roboflow
/ CVAT ile düzeltin (yanlış/eksik kutuları onarın). Ardından train/val ayırın:

```powershell
python prepare_dataset.py --split --dataset dataset --val-fraction 0.15
```

`data.yaml` bu düzeni gösterir; gerekiyorsa `path:` değerini güncelleyin.

## Adım 2 — Plaka tespit modelini eğit

```powershell
python train_detection.py --data data.yaml --epochs 100 --imgsz 640 --device 0
```

Bitince `../models/plate_detect_tr_v1.onnx` otomatik oluşur. `--imgsz` değeri
profildeki `detection.input_width/height` ile aynı olmalı (varsayılan 640).

## Adım 3 — OCR modeli

**Yol A (önce bunu deneyin) — global modeli yeniden kullan:**
```powershell
python train_ocr.py --reuse
```
Bu, çalışan global OCR modelini `plate_ocr_tr_v1.onnx` olarak kopyalar ve
uyumlu charset'i yazar. Profil hemen çalışır hale gelir.

**Yol B — kendi TR OCR modelinizi eğit (maksimum doğruluk):**
Plaka kırpıntılarını (her biri metniyle adlandırılmış, ör. `34ABC123.jpg`)
`plate_crops/train` ve `plate_crops/val` altına koyun, sonra:
```powershell
python train_ocr.py --prepare-finetune --crops plate_crops
```
Script fast-plate-ocr eğitim komutunu ve Türk alfabesini (`charset_tr.txt`)
yazdırır. Eğitim bitince ONNX'i `../models/plate_ocr_tr_v1.onnx` olarak
kopyalayın.

> TR plaka kırpıntılarını sıfırdan toplamak yerine, çalışan pipeline'ı
> `--image` / video modunda çalıştırıp tespit edilen plaka kutularını kaydederek
> hızlı bir başlangıç seti oluşturabilirsiniz (gelecek bir yardımcı script).

## Adım 4 — Devreye al

`config/anpr.json` içinde:
```json
"processing": { "active_model_profile": "turkish_finetuned", ... }
```
`turkish_finetuned` profili zaten `plate_detect_tr_v1.onnx`,
`plate_ocr_tr_v1.onnx`, `charset_tr.txt` yollarını gösteriyor. AnprStudio'dan
profil seçimi + eşik ayarlarını yapıp test edin.

## Türk plaka formatı notları

- Yapı: `<il 01-81> <1-3 harf> <2-4 rakam>` (toplam ≤ 8 karakter).
- Kullanılan harfler: A B C D E F G H I J K L M N O P R S T U V Y Z
  (Q, W, X ve Türkçe'ye özel harfler plakada yok). `charset_tr.txt` bunu yansıtır.
- Konsolidasyon katmanı (pipeline'da) aynı plakanın kare-kare titreşen
  okumalarını karakter-bazlı oylamayla tek nihai sonuca indirir; fine-tune
  doğruluğu artırır, konsolidasyon da kalan gürültüyü temizler.
