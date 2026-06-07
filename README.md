# Stuttering Detection (Kekemelik Tespiti)

Bitirme projesi: ses verilerinden elde edilen 13 MFCC katsayısına bakarak konuşma
parçalarını **Akıcı (Fluent)** veya **Kekemelik (Stutter)** olarak sınıflandıran
ML pipeline'ı. **SEP-28k** üzerinde eğitilmiş bir Random Forest modeli +
FastAPI servisi + GitHub Pages'te yayınlanan React/Vite arayüz.

🔗 Canlı demo: **https://stutter.furkan.software**

## 📂 Dizin Yapısı

```
stutter-detection/
├── api.py                          # FastAPI: /analyze, /health, CORS
├── requirements.txt                # Python bağımlılıkları
├── README.md
├── .github/workflows/deploy.yml   # GitHub Pages otomatik deploy
│
├── frontend/                       # React + Vite + Tailwind (GitHub Pages)
│
├── data/                           # CSV veri setleri
│   ├── sep28k-mfcc.csv             # SEP-28k (temizlenmiş) – sadece MFCC + is_stutter
│   ├── fluent_mfcc.csv             # VoxCeleb dev setinden üretilen akıcı örnekler
│   └── balanced_dataset.csv        # 1:1 dengelenmiş nihai eğitim seti
│
├── models/                         # Eğitilmiş model artifact'leri
│   ├── stutter_rf_model.pkl
│   └── scaler.pkl
│
├── scripts/                        # Veri hazırlama + eğitim + CLI demo
│   ├── prepare_fluent_data.py      # VoxCeleb WAV → MFCC (fluent_mfcc.csv)
│   ├── merge_datasets.py           # SEP-28k + fluent → balanced_dataset.csv
│   ├── train_model.py              # Undersampling + StandardScaler + RandomForest
│   └── live_demo_audio_analyzer.py # Yerel mikrofon CLI demo
│
├── archive/                        # Eski denemeler (jüri sunumunda "yol" anlatımı için)
└── voxceleb/                       # VoxCeleb veri seti (manuel aktarılır)
```

## 🛠️ Kullanılan Teknolojiler

- **Backend:** FastAPI, Uvicorn, librosa, scikit-learn, joblib, numpy
- **Frontend:** React 18, Vite 5, Tailwind CSS v4
- **ML:** 13 MFCC ortalaması, StandardScaler, RandomForestClassifier
- **Veri:** SEP-28k (kekemelik) + VoxCeleb dev (akıcı, gerçek veriyle dengeleme)
- **CI/CD:** GitHub Actions → GitHub Pages (özel domain: `stutter.furkan.software`)

## 📊 Veri Setleri

| Veri | Amaç | Kaynak |
|---|---|---|
| **SEP-28k MFCC** | Kekemelik etiketli (Prolongation, Block, SoundRep, WordRep, Interjection) | [Kaggle: mitulgargsam/sep28kmfcc](https://www.kaggle.com/datasets/mitulgargsam/sep28kmfcc/code) |
| **VoxCeleb1 dev** | Akıcı konuşma örnekleri (kekemelik etiketi olmayan 3 sn'lik rastgele kesitler) | [Kaggle: abdrafay1/voxceleb](https://www.kaggle.com/datasets/abdrafay1/voxceleb) |

Yerleşim:

1. SEP-28k CSV'sini `data/sep28k-mfcc.csv` olarak koy.
2. VoxCeleb1 dev setini `voxceleb/vox1_dev_wav/` altına koy (alt klasörler korunur; `rglob` ile taranır).

## 🚀 Kurulum

```bash
# 1. Repo'yu klonla
git clone <repo-url>
cd stutter-detection

# 2. Sanal ortam (Python 3.11)
python -m venv .venv311
# Windows
.venv311\Scripts\activate
# Linux / MacOS
source .venv311/bin/activate

# 3. Bağımlılıklar
pip install -r requirements.txt
```

`pyaudio` kurulumu Windows'ta bazen hata verir; wheel indirip kurmanız veya
`pipwin install pyaudio` denemeniz gerekebilir.

## 🧪 Veri Hazırlama (bir kez yapılır)

```bash
# VoxCeleb WAV'larından MFCC çıkar (akıcı veri)
python scripts/prepare_fluent_data.py
#   -> data/fluent_mfcc.csv (~3500 örnek)

# İki veri setini dengele ve birleştir
python scripts/merge_datasets.py
#   -> data/balanced_dataset.csv (1:1)
```

## 🎓 Model Eğitimi

```bash
python scripts/train_model.py
```

Çıktılar:

- `models/stutter_rf_model.pkl` (~55 MB)
- `models/scaler.pkl`
- Konsol: Accuracy, Precision, Recall, F1 raporu

Mevcut performans (Undersampling + threshold 0.625):

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Fluent | 0.88 | 0.54 | 0.67 |
| Stutter | 0.67 | 0.92 | 0.77 |
| **Accuracy** | | | **0.73** |

## 🌐 API

```bash
.venv311\Scripts\python.exe api.py
# -> http://0.0.0.0:8000
```

Endpoint'ler:

- `GET /` — servis bilgisi (threshold, chunk duration)
- `GET /health` — model hazır mı?
- `POST /analyze` — `multipart/form-data: file=<wav>` ile tahmin

CORS: Tüm origin'lere açık (GitHub Pages + ngrok uyumlu). Türkçe karakterler
UTF-8 ile düz döner (`{"label":"KEKEMELİK",...}`).

## 🖥 Frontend (GitHub Pages)

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (lokal geliştirme)
npm run build    # dist/ çıktısı
```

Deploy: `main` branch'ine push → GitHub Actions otomatik olarak
`frontend/dist` üretir, `stutter.furkan.software` domain'ine yayınlar.

İlk açılışta Backend API URL alanına ngrok veya kendi sunucunun URL'ini
yapıştırman gerekir (localStorage'da saklanır). Mobilde mikrofon erişimi için
sayfanın HTTPS üzerinden açılması şarttır.

## 🎙 CLI Demo (opsiyonel)

```bash
python scripts/live_demo_audio_analyzer.py
```

10 saniyelik mikrofon kaydı alır, `models/stutter_rf_model.pkl` ile
`/analyze` mantığını (3 sn chunk + MFCC + threshold) uygular ve zaman
çizelgesini yazdırır. Web arayüzü olmadan hızlı doğrulama içindir.

## 📝 Lisans

Bu proje eğitim ve akademik amaçlarla geliştirilmiştir.
