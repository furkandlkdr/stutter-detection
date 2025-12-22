# Stuttering Detection (Kekemelik Tespiti)

Bu proje, ses verilerinden elde edilen MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini kullanarak kekemelik tespiti yapan bir Derin Öğrenme (Deep Learning) modelidir. Bitirme projesi kapsamında geliştirilmiştir ve **SEP-28k** veri setini temel alır.

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
    *   `train_test_split`: Veriyi eğitim ve test setlerine ayırmak için.
    *   `StandardScaler`: Veriyi normalize etmek (ölçeklendirmek) için. Yapay sinir ağlarının daha hızlı ve kararlı öğrenmesi için giriş verileri standartlaştırılmıştır.
    *   `Metrics`: Model başarısını ölçmek (Confusion Matrix, Classification Report) için.
*   **TensorFlow / Keras:** Derin öğrenme modelini oluşturmak için.
    *   **Model Mimarisi:** Feed-Forward Neural Network (İleri Beslemeli Sinir Ağı).
    *   **Dense Layers:** Tam bağlantılı katmanlar.
    *   **Dropout:** Overfitting'i (aşırı öğrenme) engellemek için rastgele nöron kapatma.
    *   **Sigmoid Aktivasyonu:** Çıkış katmanında 0 ile 1 arasında bir olasılık değeri üretmek için (Binary Classification).
*   **Matplotlib & Seaborn:** Eğitim sonuçlarını (Accuracy/Loss grafikleri) ve Confusion Matrix'i görselleştirmek için.

## 🚀 Kurulum ve Çalıştırma

1.  Bu projeyi bilgisayarınıza klonlayın veya indirin.
2.  Gerekli Python kütüphanelerini yükleyin:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

3.  `sep28k-mfcc.csv` dosyasını proje dizinine yerleştirin.
4.  Modeli eğitmek ve test etmek için aşağıdaki komutu çalıştırın:

```bash
python stutter_detection.py
```

## 📊 Çıktılar

Kod çalıştırıldığında:
1.  Konsolda modelin eğitim süreci ve test sonuçları (Accuracy, Precision, Recall, F1-Score) görüntülenir.
2.  **`training_history.png`**: Eğitim ve doğrulama (validation) setleri üzerindeki Accuracy ve Loss değişimlerini gösteren grafik kaydedilir.
3.  **`confusion_matrix.png`**: Modelin tahminlerinin doğruluğunu gösteren karmaşıklık matrisi kaydedilir.

## 📝 Lisans

Bu proje eğitim ve akademik amaçlarla geliştirilmiştir.
