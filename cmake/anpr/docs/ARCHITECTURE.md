# ANPR Sistemi — Mimari Dokümanı

Katmanlı, arayüz-tabanlı, config-driven bir plaka tanıma (ANPR) embedded
sistemi. Hedef: IP kameradan (Hikvision DS-2CD1023G0E-IF) gelen video
akışını işleyip tanınan plakaları site sunucusuna TCP üzerinden iletmek.
Fiziksel kamera henüz elimizde olmadığı için tüm sistem, video dosyasından
beslenen bir simülasyon modu ile uçtan uca test edilebilir.

## 1. Genel Bakış ve Veri Akışı

```
+---------------------------------------------------------------------------+
|                        Control Layer / CLI (anpr)                         |
|   config yükleme, katmanların yaşam döngüsü, interaktif komutlar,         |
|   durum raporlama, graceful shutdown                                      |
+---------------------------------------------------------------------------+
      |  oluşturur & yönetir (factory'ler config'e bakarak seçer)
      v
+----------------+   BoundedQueue    +-----------------+   BoundedQueue   +----------------+
| Capture &      |     <Frame>       | Processing /    |  <PlateDetection>| Network /      |
| Decode Layer   | ================> | ALPR Layer      | ===============> | API Layer      |
|                |  (drop-oldest)    |                 |   (drop-newest)  |                |
| [IFrameSource] |                   | [IPlateRecognizer]                 | [INetworkTransport]
+----------------+                   +-----------------+                  +----------------+
      |                                    |                                   |
      | Simulated-    Hikvision-           | OpenCvDnn-      (ileride:         | Tcp-           (ileride:
      | FileFrame-    RtspFrame-           | PlateRecognizer  OnnxRuntime,     | Network-        MQTT, ...)
      | Source        Source(iskelet)      |                  RKNN/NPU)       | Transport
      v                                    v                                   v
  video dosyası /                    ONNX modeller                      Site sunucusu (TCP, JSON)
  RTSP stream                        (config'deki model profili)        <── komut/ayar değişikliği
```

- Her katman **kendi thread'inde** çalışır; katmanlar arası veri akışı
  sınırlı boyutlu (bounded), thread-safe kuyruklarla (`BoundedQueue<T>`,
  `core/include/core/BoundedQueue.h`) sağlanır.
- Her katman karşısındakini yalnızca **soyut arayüz** üzerinden görür;
  gerçek/simüle ayrımı üst katmanlar için görünmezdir.
- Tüm ayarlar tek bir JSON dosyasından (`config/anpr.json`) gelir; hangi
  implementasyonun kullanılacağına **factory**'ler config'e bakarak karar
  verir.

## 2. Katmanlar

### 2.1 Capture & Decode Layer (`capture/`)

**Sorumluluk:** Bir kaynaktan ham kareleri alıp `Frame` (BGR `cv::Mat` +
sıra numarası + zaman damgası + kaynak kimliği) olarak üst katmana sunmak.

**Arayüz:** `IFrameSource` (`core/include/core/interfaces/IFrameSource.h`)

| Metot                             | Açıklama                                   |
| --------------------------------- | -------------------------------------------- |
| `bool open()`                   | Kaynağı açar (dosya / RTSP).              |
| `bool getNextFrame(Frame&)`     | Sıradaki kareyi verir; EOS/hatada`false`. |
| `void close()`                  | Kaynağı serbest bırakır.                 |
| `isOpen() / id() / lastError()` | Durum ve hata raporlama.                     |

**Implementasyonlar:**

- `SimulatedFileFrameSource` — video dosyasını `cv::VideoCapture` ile okur.
  Config ile: gerçek-zaman pacing (`realtime`, `target_fps`), `frame_skip`,
  `loop`, `start_time_seconds`. Gerçek kamera gelmeden tüm pipeline'ı test
  etmenin yolu budur. (Adım 2'de implemente edilecek.)
- `HikvisionRtspFrameSource` — **UYGULANDI (2026-07-16, canlı kamerayla
  doğrulandı).** RTSP + H.265/H.264 decode (OpenCV FFmpeg backend). RTP-over-TCP
  zorlanır (`transport: "tcp"`), soket timeout ile ölü akış algılanır, kimlik
  bilgileri loglardan temizlenir (`sanitizeUrl`). `substream_url` verilirse
  analiz düşük çözünürlüklü alt akıştan yapılır. Yeniden bağlanma politikası
  kameranın capture thread'inde (açılış hatası VE akış-ortası hata aynı geri
  bağlanma döngüsüne düşer). **Karar:** özel HCNetSDK real-play yerine standart
  RTSP kullanıldı — platform bağımsızlık (Windows-only SDK DLL'leri Linux/embedded
  hedefi engellerdi), her ONVIF/RTSP markasıyla uyum, daha az bağımlılık.

`FrameSourceFactory` (`createFrameSource`) `capture.source` anahtarına göre
doğru implementasyonu örnekler.

**Kamera otomatik keşfi (`HikvisionDiscovery`, `--discover`):** Ağdaki
Hikvision cihazlarını iki yöntemle bulur — (1) SADP: UDP multicast
239.255.255.250:37020'ye XML "inquiry" probe'u, cihazlar model/seri/IP/port
ile yanıtlar (gruba katılım şart, aksi halde yanıtlar duyulmaz); (2) SADP
sessiz kalırsa (firewall/VPN yanıtları yutabilir) firewall-dostu TCP fallback:
yerel /24'lerde hem HCNetSDK portu (8000) hem RTSP (554) açık host'lar taranır,
ISAPI HTTP imzasıyla (Digest realm model adını taşır, ör. "IP Camera(AX773)")
doğrulanır. Hazır RTSP URL şablonu üretilir. AnprStudio'daki "🔍 Discover"
butonu bunu çağırıp bulunanları kamera listesine ekler.

### 2.2 Image Processing / ALPR Layer (`processing/`)

**Sorumluluk:** `Frame` alıp plaka tespiti + OCR yapmak; sonuçları
`PlateDetection` (metin, tespit/OCR güven skorları, bounding box, zaman
damgası) olarak üretmek.

**Arayüz:** `IPlateRecognizer` (`core/include/core/interfaces/IPlateRecognizer.h`)
— `initialize()`, `recognize(const Frame&) -> PlateDetectionList`.

**Motor kararı — ONNX modeller + OpenCV DNN (ilk implementasyon):**

- **OpenALPR reddedildi:** açık kaynak sürümü yıllardır bakımsız, klasik
  görüntü işleme + Tesseract tabanlı; Türk plakası için fine-tune akışı
  zayıf ve ağır bir bağımlılık zinciri getiriyor.
- **ONNX tabanlı DNN seçildi:** projenin temel gereksinimi *kendi fine-tune
  ettiğimiz Türk plakası modellerini* kod değişikliği olmadan devreye
  alabilmek. ONNX bu akışın standart değişim formatı (PyTorch/Ultralytics →
  `.onnx` export → cihaza kopyala → config'de yolu değiştir).
- **İlk çalıştırıcı olarak OpenCV'nin `dnn` modülü:** zaten bağımlılığımız
  olan OpenCV içinde geliyor (sıfır ek bağımlılık), ONNX yükleyebiliyor ve
  CPU'da yeterli. Performans gerektiğinde aynı `IPlateRecognizer` arayüzü
  arkasına ONNX Runtime (veya embedded hedefte RKNN/NPU) implementasyonu
  eklenir — üst katmanlar etkilenmez.

**Model profilleri (fine-tune desteği):** `processing.model_profiles`
altında birden fazla isimli profil tanımlanır (ör. `generic`,
`turkish_finetuned`); `processing.active_model_profile` hangisinin aktif
olduğunu seçer. Bir profil şunları içerir — hiçbiri kodda sabit değildir:

- tespit ve OCR model dosya yolları (`.onnx`),
- giriş tensor boyutları (`input_width/height`, model başına ayrı),
- ön işleme şeması (`color_order` BGR/RGB, `scale`, `mean`, `std`),
- tespit ve OCR güven eşikleri (ayrı ayrı), NMS eşiği,
- OCR karakter seti dosyası (`charset_path`) — Türk plaka alfabesi gibi
  farklı karakter setleri dosyadan okunur.

`PlateRecognizerFactory` (`createPlateRecognizer`) aktif profile ve
`engine` anahtarına bakarak doğru implementasyonu doğru parametrelerle
kurar (dependency injection).

**Varsayılan modeller (`models/`):** `generic` profili iki hazır modelle
gelir — tespit: YOLOv8n plaka dedektörü (`plate_detect_yolov8n.onnx`,
640×640 RGB/NCHW, kaynak: HuggingFace `ml-debi/yolov8-license-plate-detection`),
OCR: fast-plate-ocr global MobileViT-v2 (`plate_ocr_global_vit_v2.onnx`,
140×70 gri/NHWC, 9 slot × 37 karakter, kaynak: `ankandrew/fast-plate-ocr`).
Ön işleme farkları (renk sırası, kanal sayısı, NCHW/NHWC yerleşimi,
scale/mean/std) tamamen profil config'inden gelir — `OpenCvDnnPlateRecognizer`
YOLOv5/v8 çıktı düzenlerini otomatik ayırt eder, OCR çıktısını sabit-slot
softmax olarak decode eder (pad karakteri config'te).

**Araç-önce tespit zinciri (`vehicle_detection`, profil başına):** Plaka
dedektörü tek başına çalıştığında arka plandaki yapılarda false-positive
üretebilir ve aynı sahnede birden çok "plaka" raporlayabilir. Etkinken akış
şöyledir: önce araç nesneleri tespit edilir (COCO YOLOv8n,
`models/vehicle_detect_yolov8n.onnx`; `class_ids` ile araba/motosiklet/
otobüs/kamyon filtrelenir), sonra plaka SADECE her aracın (`roi_expand` ile
genişletilmiş) bölgesinde aranır ve araç başına EN İYİ TEK plaka adayı
alınır. Araç görülmeyen bölgelerde plaka sorgusu yapılmaz. `enabled: false`
ile eski (tüm kare) davranışa dönülür.

**Donanım hızlandırma (`dnn_backend`, profil başına):** `cpu` (varsayılan),
`opencl` / `opencl_fp16` (GPU, OpenCV'nin `opencl` feature'ı ile),
`cuda` / `cuda_fp16` (CUDA'lı özel OpenCV derlemesi gerektirir). İstenen
backend mevcut değilse sistem uyarı loglayıp CPU'ya düşer — config her
ortamda güvenle taşınabilir.

**Konsolidasyon — tek nihai sonuç (`consolidation`, `PlateTracker`):**
Geçen bir araç aynı plakayı kare-kare farklı okutur (K619879 / 1619879 /
V619879 ...). Her birini raporlamak gürültüdür; operatör nihai cevabı ister.
`PlateTracker` her kameranın okumalarını *track*'lere gruplar (metin edit
mesafesi + kutu yakınlığı), plaka görüntüden çıkınca (`finalize_after_seconds`
boşta) **tek** sonuç üretir: pozisyon-bazlı, güven-ağırlıklı karakter oylaması.
Böylece 40+ ham okuma araç geçişi başına 1 konsolide rapora iner (4K trafik
videosunda doğrulandı). `min_sightings` altındaki track'ler bastırılır;
`max_track_seconds` güvenlik sınırıdır. `enabled: false` ise eski exact-text
dedup davranışına döner.

**Deduplication (fallback):** Konsolidasyon kapalıyken, aynı plakanın tekrar
raporlanmasını önlemek için zaman-bazlı bir cache (`dedup_window_seconds`).
Her iki mekanizma da motorun *dışında*, ayrı bileşenlerde yaşar.

**Tespit görüntüsü kaydı (`detection_output`, config'ten aç/kapa):**
Raporlanan (konsolide, nihai) her plaka için işaretli bir JPG kaydedilir —
kutu + plaka metni + güven, altta zaman damgası/kamera banner'ı. Dosya adı
`<dizin>/<zaman>_<kamera>_<plaka>.jpg`. Her ham kare için değil, araç geçişi
başına **tek doğrulama görüntüsü** (tracker en yüksek güvenli kareyi saklar).
`enabled/directory/draw_timestamp/jpeg_quality` ile ayarlanır.

**Fine-tune eğitim altyapısı (`training/`):** Türk plakasına özel modelleri
üretmek için çalıştırılabilir Python boru hattı — `prepare_dataset.py`
(mevcut dedektörle zayıf-etiket bootstrap + train/val split), `train_detection.py`
(YOLOv8 fine-tune → `models/plate_detect_tr_v1.onnx`), `train_ocr.py` (global
modeli yeniden kullan **veya** fast-plate-ocr fine-tune), `charset_tr.txt`
(Türk alfabesi, Q/W/X yok). Eğitim kullanıcının GPU'sunda koşar; model
dosyaları `models/` altına düşünce `turkish_finetuned` profili kod değişmeden
devreye girer. Detaylar `training/README.md`'de.

### 2.3 Network / API Layer (`network/`)

**Sorumluluk:** `PlateDetection` verisini JSON olarak site sunucusuna
iletmek ve sunucudan gelen komut/ayar değişikliklerini alıp Control
katmanına aktarmak (çift yönlü).

**Arayüz:** `INetworkTransport` (`core/include/core/interfaces/INetworkTransport.h`)
— `start()/stop()`, `state()`, `send(payload)`, gelen mesaj ve bağlantı
durumu için handler kaydı.

**Protokol kararı — TCP (UDP değil):**

1. Plaka verisi **kayıpsız ve sıralı** ulaşmalı — UDP'de kayıp/sıra
   garantisi yok, uygulama katmanında ACK/yeniden-gönderme yazmak TCP'yi
   yeniden icat etmek olur.
2. Site tarafı cihaza **istek/komut gönderebilmeli** — kalıcı TCP bağlantısı
   üzerinden çift yönlü akış doğal; UDP'de NAT/firewall arkasındaki cihaza
   erişim sorunlu.
3. Bağlantı durumu (connected/disconnected) **izlenebilir olmalı** — TCP
   bağlantı yaşam döngüsü bunu doğrudan verir.

Transport arayüz arkasında olduğundan ileride MQTT/UDP gerekirse yalnızca
yeni bir `INetworkTransport` implementasyonu yazılır.

**Mesaj çerçevelemesi:** satır-sonu ayraçlı (newline-delimited) JSON.
Şema versiyonlanabilir (`"v": 1`). Taslak:

```json
// Cihaz -> Sunucu (plaka raporu)
{ "v": 1, "type": "plate_detection",
  "payload": { "plate": "34ABC123", "camera": "cam1",
               "detection_confidence": 0.91, "ocr_confidence": 0.87,
               "bbox": [412, 300, 180, 52],
               "timestamp": "2026-07-15T14:03:12.345Z", "frame": 1234 } }

// Cihaz -> Sunucu (heartbeat)
{ "v": 1, "type": "heartbeat",
  "payload": { "cameras_online": 8, "reports_total": 42, "timestamp": "..." } }

// Sunucu -> Cihaz (komut) — kamera bazında ALPR aç/kapat
{ "v": 1, "type": "command",
  "payload": { "name": "set_alpr", "camera": "cam1", "enabled": false } }
```

**Durum — UYGULANDI (adım 4):** `TcpNetworkTransport` kalıcı bir I/O
thread'iyle bağlanır, `reconnect_interval_seconds` ile yeniden bağlanır,
sınırlı gönderim kuyruğunu (drop-oldest; bağlantı yokken raporlar tamponlanır)
boşaltır ve gelen komutları ayrıştırıp Control'e iletir. Şema
`network/MessageSchema.h`'de; `NetworkTransportFactory` config'e göre
transport'u kurar (ileride MQTT/TLS aynı arayüzle eklenir).

**Test edilebilirlik:** `tools/mock_server` — bağımsız bir TCP test sunucusu
(`mock_server [port]`, varsayılan 9000): gelen plaka/heartbeat mesajlarını
yazdırır, stdin'den `alpr <cameraId> on|off` / `ping` komutlarını cihaza
gönderir. Uçtan uca doğrulandı (2026-07-16): canlı pipeline plakaları TCP
üzerinden versiyonlu JSON olarak sunucuya iletti, komut geri döndü.

### 2.4 Control Layer / CLI (`control/`)

**Sorumluluk:** Orkestrasyon — config'i yüklemek, CLI argümanlarını işlemek
(`--source`, `--video`, `--config`, `--log-level`), factory'lerle katmanları
kurmak, thread'leri başlatmak, interaktif stdin komutlarını (durum,
dur/başlat, config yeniden yükle) işlemek ve graceful shutdown'ı yürütmek.

**Başlatma sırası:** config → logger → network (bağlantı arka planda) →
processing → capture. **Kapanma sırası tersine:** capture durur → frame
kuyruğu kapanır → processing kuyruğu boşaltıp durur → network son mesajları
gönderip durur. Kuyrukların `close()`'u bekleyen thread'leri uyandırır;
tüm thread'ler `join` edilir.

## 3. Concurrency Modeli (capture/processing/display kısmı UYGULANDI)

```
[Capture thread] --Frame--> BoundedQueue<Frame> ----> [Processing thread]
   (kaynak fps'ine göre         (kapasite: capture.queue_capacity,        (ALPR + dedup + rapor)
    uyuyarak okur,               taşarsa EN ESKİ kare düşer —
    max_frame_width'e            bayat görüntünün değeri yok)
    küçültür)
        |
        +--Frame--> BoundedQueue<Frame> (kapasite 2) --> [Main thread / Display]
                                                          (her kareyi son tespit
                                                           overlay'i ile gösterir)

[Processing thread] --PlateDetection--> BoundedQueue<PlateDetection> --> [Network thread] (adım 4)
                             (kapasite: processing.queue_capacity,
                              taşarsa YENİ eleman reddedilir — plaka raporu kaybı loglanır)
```

Böylece yavaş bir ALPR modeli videoyu yavaşlatmaz (backlog düşürülür,
`dropped` sayacı loglanır) ve görüntüleme ALPR gecikmesinden bağımsız
kaynak fps'inde akar. Capture thread'i gerçek-zaman pacing'de uyuduğu için
CPU boşta kalır (busy-wait yok).

### 3.1 Çoklu Kamera Mimarisi (64 kameraya kadar)

Literatürdeki NVR mimarilerini (Frigate, NVIDIA DeepStream, Hailo LPR)
izler: **kamera sayısı arttıkça inference maliyeti artmaz** — sabitlenir.

```
[cam1 capture] --\
[cam2 capture] ---+--> TEK BoundedQueue<AlprJob> --> [ALPR worker havuzu]
   ...            |        (drop-oldest)               (processing.worker_count adet,
[camN capture] --/                                      her worker kendi model kopyasıyla)
     |
     |  her kamera kendi thread'inde: decode -> downscale -> MOTION GATE -> kuyruk
     v
[Grid Display] (ana thread: N×N kamera duvarı; tile tıkla = o kameranın ALPR'ı aç/kapat)
[REST API]     (ayrı thread: durum + kontrol + plaka akışı)
```

**CPU azaltma yöntemleri (hepsi config'ten):**

| Yöntem | Config anahtarı | Kaynak/gerekçe |
|---|---|---|
| Motion gating: hareket yoksa DNN hiç çalışmaz (MOG2 arka plan çıkarımı, 320px gri) | `cameras[].motion.*` | Frigate'in temel mimarisi |
| İnference havuzu: toplam DNN maliyeti kamera sayısından bağımsız, `worker_count` ile sınırlı | `processing.worker_count` | DeepStream çoklu-stream modeli |
| Kare seyreltme | `processing.process_every_n_frames` | standart |
| Erken downscale | `processing.max_frame_width` | standart |
| Araç-önce kaskad + araç başına tek plaka | `model_profiles.*.vehicle_detection` | DeepStream LPR referans uygulaması |
| OpenCV thread sınırı | `processing.num_threads` | oversubscription önleme |
| Substream analizi (kamera gelince): analiz düşük çözünürlüklü 2. stream'den | `cameras[].camera.substream_url` | NVR standardı — 64 ana stream decode edilmez |
| Donanım backend | `model_profiles.*.dnn_backend` | opsiyonel GPU |

Kuyruk dolarsa en eski iş düşer (`alpr_dropped` sayacı) — sistem hiçbir
zaman geriye yığılmaz, her zaman en güncel kareyi işler (gerçek-zamanlılık
> her kareyi işlemek).

### 3.2 REST API (`api` config bölümü)

Pipeline gömülü bir HTTP sunucusu taşır (cpp-httplib, varsayılan
`127.0.0.1:8088`):

| Endpoint | Açıklama |
|---|---|
| `GET /api/status` | toplam kare/inference/rapor sayıları, ortalama ALPR süresi, kuyruk derinliği |
| `GET /api/cameras` | kamera listesi: online/motion/ALPR durumu ve sayaçlar |
| `POST /api/cameras/<id>/alpr` gövde `{"enabled": bool}` | kamera bazında ALPR'ı çalışırken aç/kapat |
| `GET /api/plates?limit=N` | son plaka raporları (kamera, metin, güvenler, zaman) |

Bu API hem AnprStudio'nun hem de site-sunucusu entegrasyonunun kontrol
düzlemidir.

**Kimlik doğrulama (API key):** `api.api_keys` bir anahtar listesidir. Boşsa
API kimlik doğrulaması yapmaz (yalnızca `127.0.0.1`'e bağlıyken makul —
geliştirme). Dolu olduğunda **her istek** geçerli bir anahtar sunmalıdır:
`X-API-Key: <key>` ya da `Authorization: Bearer <key>` başlığı; aksi halde
`401`. Kontrol tüm endpoint'lerin önünde tek bir ön-yönlendirme kancasında
yapılır, anahtar karşılaştırması sabit-zamanlıdır. Sunucu `0.0.0.0`'a (ağa
açık) anahtar olmadan bağlanırsa yüksek sesle uyarır. AnprStudio anahtarı
config'ten okur ve canlı çağrılarında (durum/snapshot/toggle) `X-API-Key`
olarak gönderir. Dışarıya servis vermek için: `bind_address: "0.0.0.0"` +
`api_keys` doldur; TLS gerekiyorsa API'nin önüne bir reverse-proxy (nginx/
Caddy) koymak önerilir (httplib'in kendi TLS'i yerine).

- Yalnızca `std::thread` / `std::mutex` / `std::condition_variable`.
- Kuyruk taşma politikaları veri tipine göre farklı: video karesi *taze
  olan değerlidir* (drop-oldest), plaka raporu *hiç kaybolmamalıdır*
  (drop-newest + WARN log; network kuyruğu bu yüzden daha büyük).
- Shutdown: atomik `running` bayrağı + kuyrukların `close()`'u; hiçbir
  thread süresiz bloklanmaz (`waitPop` timeout'lu).

## 4. Konfigürasyon

**Format kararı — JSON (`nlohmann/json`):** Network protokolü zaten JSON
olduğundan tek serileştirme kütüphanesi iki işi de görüyor; YAML ayrı ve
daha ağır bir bağımlılık (yaml-cpp) getirirdi. `nlohmann/json` header-only
ve embedded hedefte sorunsuz.

Tek dosya: `config/anpr.json`. Tüm katman ayarları oradadır; şema
`core/include/core/Config.h` içindeki typed struct'lara birebir eşlenir ve
yükleme sırasında doğrulanır (bilinmeyen `source`, tanımsız aktif profil
vb. açıklayıcı hatayla reddedilir). CLI argümanları config'i geçersiz
kılabilir (öncelik: CLI > config dosyası > struct varsayılanları).

## 5. Klasör Yapısı

```
cmake/anpr/
├── CMakeLists.txt            # üst seviye; alt modülleri ekler
├── CMakePresets.json         # Windows (MSVC+vcpkg) ve Linux preset'leri
├── vcpkg.json                # opencv4, nlohmann-json
├── config/anpr.json          # TEK konfigürasyon dosyası
├── docs/ARCHITECTURE.md      # bu doküman
├── core/                     # ortak çekirdek (anpr_core)
│   ├── include/core/
│   │   ├── Types.h           # Frame, PlateDetection
│   │   ├── Config.h          # typed config şeması
│   │   ├── Logger.h          # seviyeli, thread-safe logger
│   │   ├── BoundedQueue.h    # bounded MPMC kuyruk
│   │   └── interfaces/       # IFrameSource, IPlateRecognizer, INetworkTransport
│   └── src/                  # Logger.cpp, Config.cpp
├── capture/                  # anpr_capture: frame kaynakları + factory
├── processing/               # anpr_processing: ALPR motorları + dedup + factory
├── network/                  # anpr_network: TCP transport + mesaj şeması
├── control/                  # anpr (executable): main, orkestrasyon, CLI
└── tests/                    # (adım 3+) birim testleri
```

Her modül kendi `CMakeLists.txt`'ine sahip bir CMake target'ıdır
(`anpr_core`, `anpr_capture`, `anpr_processing`, `anpr_network`, `anpr`).
Bağımlılık yönü daima *dışarıdan çekirdeğe* doğrudur: katman modülleri
yalnızca `anpr_core`'a (arayüzler + tipler) bağımlıdır, birbirlerine asla.

## 6. Genişletme Noktaları

| İhtiyaç                                 | Yapılacak iş                                                                           |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| Yeni kamera / kaynak                      | Yeni bir`IFrameSource` impl. + factory'ye 1 satır                                     |
| Yeni ALPR motoru (ONNX Runtime, RKNN/NPU) | Yeni bir`IPlateRecognizer` impl. + factory'ye 1 dal (`engine` anahtarı)             |
| Yeni fine-tune model versiyonu            | **Sadece config**: yeni profil ekle / `active_model_profile` değiştir, restart |
| Yeni transport (MQTT, TLS)                | Yeni bir`INetworkTransport` impl.                                                      |
| Yeni sunucu komutu                        | JSON şemaya yeni`type`/`name`, Control'de handler                                   |

## 7. Yardımcı Araçlar ve Görüntüleme

- **Canlı görüntüleme (`display` config bölümü / `--display=on|off`):**
  pipeline çalışırken işaretli (bounding box + plaka metni) video bir OpenCV
  penceresinde gösterilir; üstte fps / ALPR süresi / rapor sayısı HUD'u
  vardır, `q`/ESC ile kapanır. `max_width` ile pencere küçültülür (4K
  kaynakta önerilir). Geliştirme/doğrulama aracıdır; embedded hedefte
  kapalı tutulur.
- **CPU kontrolü:** `processing.process_every_n_frames` (ALPR'yi her N
  karede bir çalıştırır; aradaki kareler görüntülemede son tespitlerle
  çizilir), `processing.num_threads` (OpenCV iş parçacığı sınırı, 0 =
  tüm çekirdekler). Release build Debug'a göre ALPR süresini ~4× düşürür
  (4K'da ~380 ms → ~95 ms). Adım 5'teki thread'li pipeline görüntülemeyi
  ALPR'den ayırarak native fps'e çıkaracak.
- **ANPR Studio (`proj/AnprStudio`, WPF/.NET):** pipeline'ın test ve
  konfigürasyon eşlikçisi. `config/anpr.json`'ı form üzerinden düzenler
  (yalnızca gösterilen alanları değiştirir, dosyanın kalanını korur),
  `anpr.exe`'yi başlatır/durdurur, log akışını canlı gösterir ve `PLATE ...`
  satırlarını ayrıştırıp tablo + istatistik olarak sunar. Adım 4'ten sonra
  TCP üzerinden mock site sunucusu rolünü de üstlenebilir.

## 8. Platform Notları

- Kod platform-bağımsız; geliştirme şu an Windows'ta (MSVC + vcpkg,
  `CMakePresets.json`), hedef Linux/embedded (aynı preset dosyasında
  `linux-debug` mevcut).
- Ağ katmanı BSD/POSIX soket API'siyle yazılacak; Windows için ince bir
  Winsock shim dışında fark yok.
- OpenCV'nin FFmpeg backend'i her iki platformda da RTSP/H.265'i decode
  edebilir; embedded hedefte donanım decoder'a geçiş `HikvisionRtspFrameSource`
  içine izole edilmiştir.
