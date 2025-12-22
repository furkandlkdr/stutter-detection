import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. VERİ YÜKLEME VE TEMİZLEME
print("1. Veri Yükleniyor ve Temizleniyor...")
try:
    df = pd.read_csv('sep28k-mfcc.csv')
    print(f"   Orijinal veri seti boyutu: {df.shape}")
except FileNotFoundError:
    print("   HATA: 'sep28k-mfcc.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# Temizleme kriterleri: PoorAudioQuality, Music, NoSpeech, Unsure > 0 olanları çıkar
# Bu sütunların veri setinde olduğundan emin olalım
filter_cols = ['PoorAudioQuality', 'Music', 'NoSpeech', 'Unsure']
missing_cols = [col for col in filter_cols if col not in df.columns]
if missing_cols:
    print(f"   UYARI: Şu sütunlar bulunamadı, filtreleme eksik olabilir: {missing_cols}")

# Filtreleme işlemi
mask = (
    (df.get('PoorAudioQuality', 0) == 0) & 
    (df.get('Music', 0) == 0) & 
    (df.get('NoSpeech', 0) == 0) & 
    (df.get('Unsure', 0) == 0)
)

df_clean = df[mask].copy()
print(f"   Temizlenmiş veri seti boyutu: {df_clean.shape}")

# 2. ETİKETLEME (LABELING)
print("\n2. Etiketleme Yapılıyor...")
stutter_types = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']

# Hedef değişkeni oluşturma fonksiyonu
def determine_label(row):
    # Belirtilen kekemelik türlerinden herhangi biri 0'dan büyükse 1 (Stutter)
    for col in stutter_types:
        if col in row and row[col] > 0:
            return 1
    return 0

df_clean['is_stutter'] = df_clean.apply(determine_label, axis=1)

# Sınıf dağılımını göster
class_counts = df_clean['is_stutter'].value_counts()
print(f"   Sınıf Dağılımı:\n{class_counts}")
print(f"   0: Fluent (Akıcı), 1: Stutter (Kekemelik)")

# 3. ÖZELLİK SEÇİMİ (FEATURE SELECTION)
print("\n3. Özellikler Seçiliyor...")
# MFCC özellikleri: 0'dan 12'ye kadar olan sütunlar
# Sütun isimleri string '0', '1' ... '12' veya int 0, 1 ... 12 olabilir.
# CSV okurken genellikle string gelir veya int. Kontrol edelim.
mfcc_features = [str(i) for i in range(13)]

# Eğer sütunlar integer ise düzeltme yapalım
if 0 in df_clean.columns:
    mfcc_features = [i for i in range(13)]

X = df_clean[mfcc_features].values
y = df_clean['is_stutter'].values

print(f"   X (Özellikler) boyutu: {X.shape}")
print(f"   y (Hedef) boyutu: {y.shape}")

# 4. MODELLEME (KERAS/TENSORFLOW)
print("\n4. Modelleme Başlıyor...")

# Train/Test Split (%80 Train, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizasyon (Standard Scaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Mimarisi
model = Sequential([
    # Giriş katmanı (13 özellik) + 1. Gizli Katman
    Dense(64, activation='relu', input_shape=(13,)),
    Dropout(0.3), # Overfitting önlemek için
    
    # 2. Gizli Katman
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # 3. Gizli Katman
    Dense(16, activation='relu'),
    
    # Çıkış Katmanı (Binary Classification -> Sigmoid)
    Dense(1, activation='sigmoid')
])

# Modeli Derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Modeli Eğitme
print("   Model eğitiliyor...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2, # Train içinden %20 validasyon
    verbose=1
)

# 5. RAPORLAMA
print("\n5. Raporlama ve Görselleştirme...")

# Eğitim Grafikleri
plt.figure(figsize=(12, 5))

# Accuracy Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png') # Grafiği kaydet
print("   Eğitim grafikleri 'training_history.png' olarak kaydedildi.")
# plt.show() # Eğer interaktif ortamdaysanız açabilirsiniz

# Test Seti Değerlendirmesi
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n   Test Seti Doğruluğu (Accuracy): {accuracy:.4f}")
print(f"   Test Seti Kaybı (Loss): {loss:.4f}")

# Tahminler
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fluent', 'Stutter'], yticklabels=['Fluent', 'Stutter'])
plt.title('Confusion Matrix')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.savefig('confusion_matrix.png')
print("   Confusion Matrix 'confusion_matrix.png' olarak kaydedildi.")

# Classification Report
print("\n   Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Fluent', 'Stutter']))

print("\nİşlem Tamamlandı.")
