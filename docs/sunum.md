# Bitirme Projesi: Akustik Kekemelik Tespit Sistemi

> **Sunum formatı:** Gamma.app uyumlu Markdown
> **Hedef kitle:** Bitirme jürisi + akranlar
> **Tahmini süre:** 10 dakika
> **Stil:** Mühendislik odaklı, kanıta dayalı, hikâye anlatımı

---

## Slayt 1 — Projenin Amacı ve Kapsamı

### Akustik kekemelik tespit sistemi: Sese kulak vermek

- **Amaç:** Gerçek zamanlı, dilden bağımsız bir **akustik kekemelik tespit** sistemi geliştirmek.
- **Yaklaşım:** Metin veya dil bilgisi değil, **salt ses sinyali** üzerinden çalışır.
- **Neden ses?** Kekemelik fizyolojik bir olgudur; ses yolundaki kas gerginliği, tekrarlar ve uzatmalar **akustik iz** bırakır. Bu iz dil bağımsızdır.
- **Çıktı:** Bir ses kaydını alıp saniyeler içinde "Kekemelik var / yok" etiketi üreten bir **uçtan uca (end-to-end) sistem**.

> *Speaker note: "Bu proje bir dil tanıma problemi değil, bir sinyal işleme ve sınıflandırma problemi."*

---

## Slayt 2 — İlk Zorluk: SEP-28k Veri Seti

### Başlangıç noktası: SEP-28k

- **Veri kaynağı:** [SEP-28k](https://www.kaggle.com/datasets/aryansinghal102/stuttering-detection-dataset) — 28 saat, 4 podcast'ten yaklaşık **38 bin** 3 saniyelik kırpma.
- **Etiket yapısı:** Veri seti kekemeliği 7 alt sınıfa ayırır (Block, Prolongation, SoundRep, WordRep, Interjection, NoStutter, NoSpeech).
- **Birleştirme kararı:** Alt sınıfları tek bir `is_stutter=1` sütununa indirgedik. Amaç, **genel bir kekemelik dedektörü** eğitmek.
- **Özellik çıkarımı:** Ham dalga formu yerine **13 MFCC** (Mel-Frequency Cepstral Coefficients) vektörü. Neden MFCC? Konuşma tanımada altın standart; insan kulağının frekans algısını taklit eder.
- **Kritik eksik:** SEP-28k'ta **akıcı konuşma örnekleri son derece az**.

> *Speaker note: "Eğer model yalnızca kekemelik görürse, kekemelikten başka bir şey tanımayı öğrenemez."*

---

## Slayt 3 — İlk Deneme: SMOTE ve Hüsran

### Sentetik veri ile kurtarma girişimi

- **Sorun:** Sınıf dengesizliği — kekemelik örnekleri baskın, akıcı konuşma yok denecek kadar az.
- **İlk çözüm:** **SMOTE** (Synthetic Minority Over-sampling Technique) ile sentetik akıcı örnekler üreterek denge sağlamak.
- **Beklenti:** Dengesiz veride ezilen azınlık sınıfı (akıcı) güçlenecek, model her iki sınıfı da öğrenecek.
- **Gerçek:**
  - Accuracy: %95+
  - **Fluent Recall: %0** (tahmin edilenlerden neredeyse hiçbiri doğru değildi)
  - **False Positive oranı:** Patladı
- **Teşhis:** Model, güvenli yol olarak **her sesi "kekemelik"** olarak etiketliyordu. Sentetik veri gerçek konuşma dağılımını yansıtmıyordu.

> *Speaker note: "Yüksek accuracy aldatıcıdır. Dengesiz veride accuracy, modelin ne kadar başarılı olduğunu değil, baskın sınıfı ne kadar iyi ezberlediğini ölçer."*

---

## Slayt 4 — Çözüm Arayışı: Yeni Veri Seti Entegrasyonu

### Gerçek akıcı konuşma: VoxCeleb

- **Karar:** Sentetik veri üretmek yerine **gerçek insan sesi** toplamak.
- **Aday 1 — LibriSpeech:** Kitap okuma kayıtları. Çok temiz, çok kontrollü, **gerçek hayata uzak**.
- **Seçim — VoxCeleb:** YouTube'dan derlenmiş doğal konuşma verisi. Fısıltılar, duraklamalar, doğal hız değişimleri içerir.
- **İşlem adımları:**
  1. VoxCeleb `vox1_dev_wav` setinden örnekler seçildi.
  2. 16 kHz mono'ya dönüştürüldü, 3 saniyelik pencerelere bölündü.
  3. Her pencereden **13 MFCC ortalaması** çıkarıldı.
  4. `is_stutter=0` etiketiyle kaydedildi.
- **Sonuç:** **~7.905 gerçek akıcı örnek** eklendi.

> *Speaker note: "Modele 'bu da kekemelik değil' demeyi öğretmek için, gerçek hayattan konuşma örnekleri şart."*

---

## Slayt 5 — Dengesiz Veri ve Undersampling Çözümü

### Organik denge: 1:1 altın oran

- **Yeni dağılım:**
  - Kekemelik: 16.305
  - Akıcı: 7.905
  - Oran: **1:2.06** (hâlâ dengesiz)
- **Dengesiz kaldığında sonuç:**
  - Accuracy: %81.76 (yanıltıcı yüksek)
  - **Fluent Recall: %44** — her 2 akıcı konuşmadan 1'i yanlış etiketleniyor
- **Çözüm — Undersampling (Alt Örnekleme):** Çoğunluk sınıfını (kekemelik) rastgele seçimle akıcı sınıfa eşitle.
- **Final dağılım:**
  - Kekemelik: 7.905
  - Akıcı: 7.905
  - Oran: **1:1**
- **Kazanç:** Sentetik veri yok, gerçek dağılım korundu, model her iki sınıfı da eşit ağırlıkla gördü.

> *Speaker note: "Undersampling, veri kaybı gibi görünür ama modelin öğrenme kalitesi için gerekli bir fedakârlıktır."*

---

## Slayt 6 — Model Seçimi ve Optimizasyon

### Random Forest, XGBoost ve LightGBM karşılaştırması

- **Aday modeller:** Üç güçlü ağaç-tabanlı sınıflandırıcı aynı eğitim seti üzerinde koşuldu.
- **Karşılaştırma metrikleri:** Sınıf bazlı F1 + Macro F1

| Model | Stutter F1 | Fluent F1 | Macro F1 |
|-------|-----------|-----------|----------|
| **Random Forest** | **0.774** | 0.666 | **0.720** |
| XGBoost | 0.734 | 0.670 | 0.702 |
| LightGBM | 0.734 | 0.672 | 0.703 |

- **Seçim gerekçesi:** Random Forest hem en yüksek **Macro F1**'e hem de en yüksek **Stutter F1**'e sahip.
- **Ek kazanımlar:** Daha düşük kütüphane bağımlılığı, yorumlanabilir feature importance, hızlı inference.
- **Eşik (threshold) optimizasyonu:** Varsayılan `0.50` yerine **0.625** seçildi.
  - **Fluent Recall:** %54 → %67.8
  - **False Positive:** %8 → %4
  - Amaç: Doğal konuşmadaki duraklamaları (hesitation) kekemelik olarak etiketlememek.

> *Speaker note: "Eşik optimizasyonu, modelin 'şüpheli' bölgesinde nasıl davranacağını ayarlar. Bizim için yanlış alarm vermemek, bazen keşke kaçırmaktan daha değerli."*

---

## Slayt 7 — Sistem Mimarisi ve API Entegrasyonu

### Ürünleşme: FastAPI + Ngrok

- **Backend:** Python 3.11 + **FastAPI** + Uvicorn.
- **Endpoint'ler:**
  - `GET /health` — sağlık kontrolü
  - `POST /analyze` — ses dosyasını alır, MFCC çıkarır, model ile tahmin yapar, JSON döner
  - `OPTIONS /{full_path}` — CORS preflight (web için zorunlu)
- **Tahmin akışı:**
  1. Ses alınır (WAV, 16 kHz mono)
  2. 3 saniyelik pencerelere bölünür
  3. Her pencere → 13 MFCC ortalaması
  4. StandardScaler ile normalizasyon
  5. Random Forest ile `is_stutter` tahmini
  6. Sonuçlar dalga formu (waveform) verisi ile birlikte döner
- **Dış dünya erişimi:** Modem port yönlendirmesi (port forwarding) yerine **Ngrok** tüneli. Güçlü yerel makine, tek komutla HTTPS üzerinden erişilebilir hale gelir.
- **Neden Ngrok?** NAT/CGNAT arkasındaki ev internetlerinde port açmak zor/güvensiz; Ngrok bu sorunu şifreli tünelle çözer.

> *Speaker note: "Akademik bir modelden ürüne geçişin en kritik adımı: API sözleşmesini net tanımlamak."*

---

## Slayt 8 — Son Kullanıcı Arayüzü

### Frontend: Vite + React + Tailwind + PWA

- **Stack:**
  - **Vite** — hızlı geliştirme ortamı ve build aracı
  - **React 18** — bileşen tabanlı UI
  - **Tailwind CSS v4** — utility-first stil sistemi
  - **MediaRecorder API + AnalyserNode** — tarayıcıda mikrofon kaydı ve canlı dalga formu
- **Kullanıcı deneyimi:**
  - **Dosya yükle** veya **mikrofonla kaydet** seçenekleri
  - Kayıt sırasında gerçek zamanlı dalga formu animasyonu
  - Sonuç ekranında: toplam süre, kekemelik anları zaman çizelgesi, dalga formu üzerinde renkli vurgular
  - **Mobil uyumlu** — telefonda da rahat kullanım
- **PWA (Progressive Web App):**
  - `manifest.json` + service worker ile "ana ekrana ekle" desteği
  - Offline çalışma kabiliyeti
  - Uygulama hissi (native app benzeri)
- **Yayınlama:** **GitHub Actions** ile her push'ta otomatik build → **GitHub Pages**'e deploy.
  - Özel alan adı: **stutter.furkan.software**

> *Speaker note: "Sıfır sunucu maliyetiyle, CI/CD hattıyla otomatik yayınlanan bir PWA — üretim kalitesinde bir mimari."*

---

## Slayt 9 — Canlı Demo ve Gelecek Çalışmalar

### Bugün ve yarın

- **Şu an:**
  - Sistem uçtan uca çalışıyor: mikrofon → API → sonuç ekranı
  - **stutter.furkan.software** üzerinden canlı erişim
  - 24.210 satır eğitim verisi + doğrulanmış 4 metrik grafiği
- **Doğrulanan çıktılar:**
  - Macro F1: **0.720**
  - Stutter F1: **0.774**
  - Fluent Recall: **%67.8** (eşik 0.625 ile)
- **Demo akışı:**
  1. Site açılır
  2. Mikrofon izni verilir
  3. 3 saniyelik ses kaydı alınır
  4. Sonuç 1 saniyenin altında döner

#### Gelecek vizyonu

- **IoT entegrasyonu:** Akustik motor, giyilebilir stres/nabız sensörleriyle birleştirilerek **stres-kekemelik korelasyonu** araştırmalarına zemin hazırlayabilir.
- **Kişiselleştirme:** Kullanıcıya özel ince ayar (fine-tuning) ile baseline'lar oluşturma.
- **Çok dilli doğrulama:** Sistemin dil bağımsızlığını akademik olarak kanıtlamak için farklı dil veri setleriyle test.
- **Edge deployment:** ONNX / TensorFlow Lite dönüşümüyle mobilde offline çalışma.

> *Speaker note: "Bu çalışma bir bitirme projesi ama akustik motoru bir platform — üzerine inşa edilecek çok şey var."*

---

## Ek — Teknik Özet

### Pipeline özeti

```
Ses kaydı (WAV, 16 kHz mono)
    ↓
3 sn pencerelere böl
    ↓
13 MFCC ortalaması (librosa / saf numpy)
    ↓
StandardScaler (eğitim istatistikleri)
    ↓
RandomForestClassifier(n_estimators=200)
    ↓
Threshold 0.625 ile sınıflandır
    ↓
is_stutter ∈ {0, 1}
```

### Metrikler (test seti, %20 stratified split)

| Metik | Undersampling öncesi | Undersampling sonrası | + Eşik 0.625 |
|-------|---------------------|-----------------------|--------------|
| Accuracy | 0.818 | 0.730 | — |
| Fluent Recall | 0.44 | 0.54 | **0.678** |
| Fluent F1 | 0.61 | 0.67 | — |
| Stutter F1 | 0.88 | 0.77 | — |
| **Macro F1** | — | 0.720 | — |

### Repolar ve kaynaklar

- **Proje:** [github.com/furkan-software/stutter-detection](https://github.com)
- **Veri:** [SEP-28k](https://www.kaggle.com/datasets/aryansinghal102/stuttering-detection-dataset) · [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- **Canlı:** [stutter.furkan.software](https://stutter.furkan.software)
