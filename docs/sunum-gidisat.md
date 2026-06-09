# 🎤 Sunum Planı — Stuttering Detection

> Toplam süre: **9 dakika 30 saniye** (10 dakika sınırının altında).
> Sunum 5 sahneden oluşur; her sahne için ne söyleneceği, ne gösterileceği ve ne kadar süre harcanacağı yazıyor.

---

## ⏱️ Zamanlama (özet)

| # | Sahne | Süre | Hedef |
|---|---|---|---|
| 1 | Açılış & problem | 0:30 | Jürinin ilgisini çek |
| 2 | Proje mimarisi | 1:30 | Büyük resmi göster |
| 3 | Dosya turu | 1:30 | Kodun kalitesini hissettir |
| 4 | Canlı demo | 4:00 | Şov; jüri kendi gözüyle görsün |
| 5 | Sonuç & sorular | 2:00 | Kapanış + Q&A |

---

## 🎬 Sahne 1 — Açılış & Problem (0:30)

**Ekranda:** Slayt başlığı — *"Stuttering Detection — Bitirme Projesi"*. Sade, NAFAIR logosu, isim.

**Söyle:**

> "Ses verisinden kekemelik tespit eden, gerçek zamanlı çalışan bir yapay zekâ sistemi geliştirdim. Problem, dünyada ~70 milyon insanın yaşadığı, Türkiye'de ise her 100 çocuktan 4'ünde görülen kekemelik. Amaç: bir kayıt alındığında, 3 saniyelik pencereler halinde bu konuşmada kekemelik olup olmadığını otomatik olarak gösterebilmek."

**Kapanış cümlesi:** "Sistemimizi uçtan uca göstereyim — 9 dakika sürecek."

---

## 🏗️ Sahne 2 — Mimari (1:30)

**Ekranda:** [README.md](README.md) içindeki dizin ağacı (`stutter-detection/`).

**Söyle (3 katmanı sırayla anlat):**

> "Üç katmanlı bir sistem kurdum. **Veri**: SEP-28k kekemelik veri seti, üzerine VoxCeleb'den gerçek akıcı konuşma örnekleri ekleyerek 1:1 organik denge sağladım — sentetik veri üretmedim. **ML çekirdeği**: 13 MFCC katsayısının ortalamasını alıyorum, StandardScaler ile ölçekliyorum, Random Forest ile sınıflandırıyorum. **Servis katmanı**: FastAPI ile `/analyze` endpoint'i yayınlıyorum, frontend GitHub Pages'te barınıyor — yani API'yi başka bir bilgisayara açıp frontend oradan konuşuyor."

**Vurgu:** SMOTE, class weighting gibi yapay yöntemleri bilinçli olarak reddettik, gerçek veriyle dengeledik — bu jüri için önemli bir tasarım kararı.

---

## 🗂️ Sahne 3 — Dosya Turu (1:30)

**Ekranda:** VS Code, dosya gezgini açık.

**Gösterilecek dosyalar (her biri 15-20 sn):**

1. **Dizin yapısı** ([README.md](README.md) zaten gösterildi):
   - `data/`, `models/`, `scripts/`, `frontend/`, `api.py` — her klasörün amacı tek cümle.

2. **[scripts/train_model.py](scripts/train_model.py)** — model eğitimi:
   - "Undersampling burada, 1:1 denge, sadece StandardScaler + Random Forest. Karmaşık bir şey yok; ama threshold 0.625 — bunu F1 eğrisinden bulduk."

3. **[api.py](api.py)** — servis:
   - "/analyze endpoint'i: 3 sn'lik chunk, MFCC, scaler, model, threshold — tüm pipeline burada. 60 çubukluk bir waveform de dönüyor ki frontend görsel gösterebilsin."

4. **[frontend/src/App.jsx](frontend/src/App.jsx)** — arayüz:
   - "React + Vite + Tailwind. API URL'ini kullanıcı giriyor, localStorage'da saklanıyor. Canlı mikrofon kaydı + dosya yükleme + analiz + sonuç kartları + yatay timeline + dalga formu — hepsi tek dosyada."

**Kapanış:** "Şimdi canlıya geçelim."

---

## 🎙️ Sahne 4 — Canlı Demo (4:00)

**Ekranda:** `https://stutter.furkan.software` (GitHub Pages).

### 4a. Site açılışı (0:30)

- Sayfayı taze sekmede aç.
- Sağ üstte sağlık rozetini göster: **"API hazır ✓"** (yeşil tik). Bu otomatik kontrol backend'in ayakta olduğunu ispatlar.
- Kısa cümle: "Site HTTPS üzerinden yayında, API'si kendi bilgisayarımda ngrok üzerinden açık."

### 4b. Dosya yükleme (1:30)

- "Elimde SEP-28k'ten aldığım, üzerinde bilinen kekemelik etiketleri olan örnekler var."
- `archive/test_session.wav` veya kısa bir örnek yükle.
- **Analiz Et**'e bas.
- Sonuç geldiğinde **parça parça** göster:
  - "Toplam Süre kartı, Tespit Edilen Kekemelik sayısı — bu örnekte X tane."
  - **Yatay Zaman Çizelgesi**: "Kırmızılar kekemelik tespit edilen 3 saniyelik pencereler, yeşiller akıcı. Üstteki dalga formu da sesin gerçek enerjisini gösteriyor — yoğun yerlerde çubuklar yükseliyor."
  - **Parça Listesi**: "Her chunk için confidence yüzdesi. Eşik değerimiz 0.625 — onun üzerine çıkanlar KEKEMELİK, altında kalanlar AKICI."

### 4c. Canlı mikrofon (1:30)

- "Şimdi de web arayüzünden mikrofonla anlık kayıt yapıp aynı sisteme gönderelim."
- **Mikrofondan Kaydet**'e bas. (Sayfa HTTPS üzerinden açık olduğu için tarayıcı izin isteyecek — "İzin ver"e bas.)
- Konuş: "Merhaba, b-b-ben b-bugün s-s-sınıfta k-k-konuştum" gibi bilinçli kekemelik.
- **Dur**'a bas. Ekranda "WAV'a dönüştürülüyor…" mesajı görünecek (tarayıcıda dönüşüm anı).
- **Analiz Et**'e bas.
- Sonuçlar geldiğinde: "Gördüğünüz gibi gerçek zamanlı kayıttan da çalışıyor. Mikrofon simgesinin pulse animasyonu sesle birlikte büyüyüp küçülüyordu — mobilde de net görünüyor."

### 4d. Vurgu (0:30)

- "Tek bir threshold sayesinde Fluent Recall'ı %44'ten %54'e çıkardık, F1'i 0.61'den 0.67'ye. SMOTE gibi sentetik yöntemlere gerek kalmadan, gerçek akıcı veri ekleyerek."
- "Hepsi GitHub Pages'te yayında, mobilde HTTPS üzerinden çalışıyor. CI/CD pipeline'ı main branch'e her push'ta otomatik deploy."

---

## 🎯 Sahne 5 — Sonuç & Q&A (2:00)

### Kapanış cümlesi (0:30)

> "Özetle: gerçek veriyle dengelenmiş bir ML modeli, FastAPI üzerinden dışarıya açılmış bir servis, ve GitHub Pages'te yayınlanan mobil-uyumlu bir arayüz. Ses dosyası yükleyin veya mikrofonla kayıt yapın, sistem 3 saniyelik pencereler halinde kekemelik tespit ediyor. Teşekkürler, sorularınızı alabilirim."

### Olası Jüri Soruları & Cevaplar (1:30)

**S: Neden SMOTE veya class weighting kullanmadınız?**
> "Sentetik veri üretmek, modelin gerçek dağılımdan uzaklaşmasına yol açıyor. Bunun yerine gerçek akıcı konuşma verisi topladık — VoxCeleb'den 3500 örnek. Undersampling ile 1:1 denge sağladık. Bu daha dürüst ve tekrarlanabilir bir yöntem."

**S: Threshold 0.625 nereden geldi?**
> "F1 eğrisinden, Fluent F1'ini maksimize eden değer olarak çıktı. Deneysel; farklı threshold'lar denenerek bu noktada Fluent Recall'ı %10 artırdık (44% → 54%) — fakat toplam accuracy biraz düştü. Sınıflar arası denge için bu trade-off'u kabul ettik."

**S: 13 MFCC yeterli mi? Daha fazla özellik denediniz mi?**
> "İlk denemelerde delta, delta-delta, prosody (pitch, energy) gibi ek özellikleri de denedik. Random Forest'ın bu küçük boyutlu (13) feature set'inde en iyi performansı verdiğini gördük. Daha derin modeller (XGBoost, LightGBM) denenmesine rağmen macro F1 anlamlı fark yaratmadı — MFCC + RF yeterince iyi."

**S: Gerçek zamanlı / streaming yapıyor musunuz?**
> "Şu an dosya tabanlı. Tarayıcı mikrofonu zaten canlı kayıt yapıyor, kayıt bitince API'ye gönderiyoruz. Gerçek streaming için WebSocket tabanlı bir chunk-streaming endpoint'i eklenebilir — backend mimarisi buna uygun, sadece frontend'de MediaRecorder'ı parça parça gönderecek şekilde değiştirmek gerekir."

**S: Veri setleri ne kadar büyük?**
> "SEP-28k: ~28 bin etiketli klib; bizim kullandığımız temizlenmiş hali ~16 bin stutter örneği. VoxCeleb: dev seti ~148 bin ses dosyası, biz 3500 rastgele 3 sn'lik kesit aldık. Toplam eğitim setimiz dengeleme sonrası ~16 bin satır."

**S: Türkçe için ne kadar uyumlu?**
> "Model akustik özelliklere (MFCC) bakıyor, dile bağımlı değil. Türkçe konuşan kişilerin kayıtlarını da doğrudan işleyebilir. Ancak kekemelik tipi dağılımı dile/kültüre göre değişebileceğinden, Türkçe'ye özel bir fine-tune yapılabilir."

**S: Deploy nasıl çalışıyor?**
> "GitHub Actions: main branch'e push olunca, frontend/dist'i alıp GitHub Pages'e yayınlıyor. CNAME dosyasıyla `stutter.furkan.software` domain'ine yönlendiriliyor. API ise kendi bilgisayarımda uvicorn olarak çalışıyor, ngrok üzerinden public URL alıyor."

**S: Projeyi nasıl geliştirmeye devam edeceksiniz?**
> "Kısa vadede: WebSocket streaming, Türkçe veriyle fine-tune. Uzun vadede: prosody ve dilbilgisi özellikleri ekleyerek daha derin bir model, gerçek klinik vakalarla değerlendirme, dil terapistine yönlendiren bir asistan arayüzü."

---

## ✅ Demo Öncesi Checklist

Sunumdan **1 saat önce** şunları doğrula:

### Backend
- [ ] `.venv311` aktif, `pip install -r requirements.txt` çalıştırıldı
- [ ] `models/stutter_rf_model.pkl` ve `models/scaler.pkl` yerinde (~55 MB)
- [ ] `python api.py` çalışıyor, logda "Model yüklendi" mesajı var
- [ ] `curl http://localhost:8000/health` → `{"ready": true}`

### Public erişim
- [ ] ngrok (veya cloudflared) çalışıyor, URL: `https://xxx.ngrok-free.dev`
- [ ] ngrok URL'ini tarayıcıda aç, sağlık rozeti **yeşil tik** gösteriyor
- [ ] Telefonundan da aç, HTTPS kilit ikonu görünüyor

### Frontend (GitHub Pages)
- [ ] `https://stutter.furkan.software` açılıyor
- [ ] Sayfada **"Stutter Detection · Nafair - 2026"** başlığı görünüyor
- [ ] API URL alanına ngrok URL'ini yapıştır, **Kaydet**
- [ ] Sağlık rozeti **yeşil tik** veriyor

### Demo materyalleri
- [ ] `archive/test_session.wav` (bilinen kekemelik etiketli örnek) hazır
- [ ] Varsa bir tane kısa "akıcı" örnek (karşılaştırma için)
- [ ] Mikrofon izin diyaloğu için: "İzin ver" diyeceksin
- [ ] Konuşma örneğin hazır: "Merhaba, b-b-ben bugün s-s-sınıfta konuştum"

### Yedek plan
- [ ] Demo başarısız olursa: `archive/stutter_detection.py` + eski confusion_matrix.png'ler gösterilebilir
- [ ] İnternet kesilirse: localhost'ta `python api.py` + `npm run dev` çalıştır

---

## 🎯 Sunum Sonu Kapanış

> "Projeyi açık kaynak olarak GitHub'da paylaştım, README'de tüm kurulum adımları var. Katkı ve geri bildirim almaktan memnuniyet duyarım. Tekrar teşekkürler."

Jürinin alkışı veya soruları → doğal Q&A akışına geç.
