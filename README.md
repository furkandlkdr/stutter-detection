# Stuttering Detection (Kekemelik Tespiti)

Bu proje, ses verilerinden elde edilen MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini kullanarak kekemelik tespiti yapan bir Makine Öğrenmesi (Machine Learning) modelidir. Bitirme projesi kapsamında geliştirilmiştir ve **SEP-28k** veri setini temel alır.

## 🎯 Proje Amacı

Projenin temel amacı, verilen ses kesitlerinin MFCC özelliklerine bakarak, konuşmanın **Akıcı (Fluent)** mı yoksa **Kekemelik (Stutter)** içerip içermediğini sınıflandırmaktır. Bu çalışma, ileride geliştirilecek olan "Anlık Ses ile Kekemelik Tespiti" sisteminin temelini oluşturmaktadır.

## 📂 Veri Seti

Bu projede kullanılan veri seti **SEP-28k** (Stuttering Events in Podcasts) veri setinden türetilmiştir. Model, ham ses dosyaları yerine, bu seslerden çıkarılmış **MFCC** özelliklerini kullanır.

Veri setine aşağıdaki linkten ulaşabilirsiniz:
🔗 [SEP-28k MFCC Dataset - Kaggle](https://www.kaggle.com/datasets/mitulgargsam/sep28kmfcc/code)

**Not:** İndirdiğiniz `sep28k-mfcc.csv` dosyasını projenin ana dizinine atmanız gerekmektedir.

## 🛠️ Kullanılan Teknolojiler ve Yöntemler

Bu projede aşağıdaki kütüphaneler ve yöntemler kullanılmıştır:

*   **Python 3.x:** Projenin ana programlama dili.
*   **Pandas:** Veri manipülasyonu, CSV okuma ve veri temizleme işlemleri için.
*   **NumPy:** Sayısal işlemler ve dizi (array) manipülasyonları için.
*   **Scikit-Learn:**
    *   `RandomForestClassifier`: Sınıflandırma modeli olarak kullanılmıştır.
    *   `train_test_split`: Veriyi eğitim ve test setlerine ayırmak için.
    *   `Metrics`: Model başarısını ölçmek (Confusion Matrix, Classification Report) için.
*   **Librosa & PyAudio:** Ses işleme ve kayıt işlemleri için.
*   **Joblib:** Eğitilen modeli kaydetmek ve yüklemek için.
*   **Matplotlib & Seaborn:** Sonuçları görselleştirmek için.

## 🚀 Kurulum ve Çalıştırma

1.  Bu projeyi bilgisayarınıza klonlayın veya indirin.
2.  Sanal bir Python ortamı oluşturun ve gerekli paketleri yükleyin:

```bash
python -m venv .venv311
# Windows
.venv311\Scripts\activate
# Linux/MacOS
source .venv311/bin/activate
pip install -r requirements.txt
```

**Not:** `pyaudio` kurulumunda hata alırsanız, işletim sisteminize uygun `PyAudio` wheel dosyasını indirip kurmanız gerekebilir veya `pipwin install pyaudio` deneyebilirsiniz.

3.  `sep28k-mfcc.csv` dosyasını proje ana dizinine yerleştirin.

### Adım 1: Modeli Eğitme

Canlı tespit sistemi `live-detection` klasörü altında çalışmaktadır. Önce bu klasöre girip modeli eğitmelisiniz:

```bash
cd live-detection
.venv311\Scripts\python.exe train_model.py
```
Bu işlem, ana dizindeki veri setini okuyacak ve `live-detection` klasörü içinde `stutter_rf_model.pkl` ve `scaler.pkl` dosyalarını oluşturacaktır.

### Adım 2: Ses Analizi (Canlı Kayıt)

Model eğitildikten sonra, yine `live-detection` klasörü içindeyken mikrofonunuzu kullanarak analiz yapabilirsiniz:

```bash
python audio_analyzer.py
```
Bu script, varsayılan olarak 10 saniyelik bir ses kaydı alır (kod içinden değiştirilebilir), bunu 3'er saniyelik parçalara böler ve her parça için kekemelik analizi yapar.

## 📊 Çıktılar

*   **Eğitim:** Accuracy, Precision, Recall, F1-Score metrikleri ve grafikler.
*   **Analiz:** Zaman damgalı (Timestamped) kekemelik çizelgesi. Örn:
    ```text
    0.0s - 3.0s   | AKICI           | %85.0
    3.0s - 6.0s   | KEKEMELİK       | %72.4
    ```

## 📝 Lisans

Bu proje eğitim ve akademik amaçlarla geliştirilmiştir.
