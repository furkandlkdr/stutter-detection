import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# SMOTE kütüphanesi kontrolü
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("HATA: 'imbalanced-learn' kütüphanesi bulunamadı.")
    print("Lütfen terminalde şu komutu çalıştırın: pip install imbalanced-learn")
    exit()

# 1. VERİ YÜKLEME VE TEMİZLEME
print("1. Veri Yükleniyor ve Temizleniyor...")
try:
    df = pd.read_csv('sep28k-mfcc.csv')
    print(f"   Orijinal veri seti boyutu: {df.shape}")
except FileNotFoundError:
    print("   HATA: 'sep28k-mfcc.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# Temizleme kriterleri: PoorAudioQuality, Music, NoSpeech, Unsure > 0 olanları çıkar
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

def determine_label(row):
    for col in stutter_types:
        if col in row and row[col] > 0:
            return 1
    return 0

df_clean['is_stutter'] = df_clean.apply(determine_label, axis=1)

# Sınıf dağılımını göster
class_counts = df_clean['is_stutter'].value_counts()
print(f"   Sınıf Dağılımı (Orijinal):\n{class_counts}")
print(f"   0: Fluent (Akıcı), 1: Stutter (Kekemelik)")

# 3. ÖZELLİK SEÇİMİ (FEATURE SELECTION)
print("\n3. Özellikler Seçiliyor...")
# MFCC özellikleri: 0'dan 12'ye kadar olan sütunlar
mfcc_features = [str(i) for i in range(13)]

# Eğer sütunlar integer ise düzeltme yapalım
if 0 in df_clean.columns:
    mfcc_features = [i for i in range(13)]

X = df_clean[mfcc_features].values
y = df_clean['is_stutter'].values

# 4. TRAIN/TEST SPLIT
print("\n4. Train/Test Ayrımı Yapılıyor (%80 Train, %20 Test)...")
# Stratify=y diyerek her iki sette de oranların korunmasını sağlıyoruz (ilk başta)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Eğitim Seti Boyutu (SMOTE öncesi): {X_train.shape}")
print(f"   Test Seti Boyutu: {X_test.shape}")

# 5. VERİ DENGELEME (SMOTE)
print("\n5. SMOTE ile Eğitim Verisi Dengeleniyor...")
# SMOTE sadece eğitim setine uygulanır! Test setine dokunulmaz (Data Leakage önlemek için).
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"   Eğitim Seti Boyutu (SMOTE sonrası): {X_train_resampled.shape}")
print(f"   Yeni Sınıf Dağılımı (Train): {pd.Series(y_train_resampled).value_counts().to_dict()}")

# 6. MODELLEME (RANDOM FOREST)
print("\n6. Random Forest Modeli Eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, y_train_resampled)

# 7. RAPORLAMA
print("\n7. Raporlama ve Görselleştirme...")

# Tahminler
y_pred = rf_model.predict(X_test)

# Metrikler
acc = accuracy_score(y_test, y_pred)
print(f"   Test Seti Doğruluğu (Accuracy): {acc:.4f}")
print("\n   Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Fluent', 'Stutter']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Fluent', 'Stutter'], yticklabels=['Fluent', 'Stutter'])
plt.title('Confusion Matrix (Random Forest)')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png')
print("   Confusion Matrix 'confusion_matrix_rf.png' olarak kaydedildi.")

# Feature Importance (Özellik Önemi)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f"MFCC-{f}" for f in mfcc_features]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (MFCC Özelliklerinin Önemi)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='teal')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.xlabel("MFCC Katsayıları")
plt.ylabel("Önem Derecesi")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("   Feature Importance grafiği 'feature_importance.png' olarak kaydedildi.")

print("\nİşlem Tamamlandı.")

