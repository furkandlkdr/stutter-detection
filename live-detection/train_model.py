import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# SMOTE kütüphanesi kontrolü
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("HATA: 'imbalanced-learn' kütüphanesi bulunamadı.")
    print("Lütfen terminalde şu komutu çalıştırın: pip install imbalanced-learn")
    exit()

def train_and_save_model():
    print("1. Veri Yükleniyor ve Temizleniyor...")
    # Veri seti ana klasörde olabilir, kontrol edelim
    dataset_path = 'sep28k-mfcc.csv'
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join('..', 'sep28k-mfcc.csv')
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"   Veri seti yüklendi: {dataset_path}")
    except FileNotFoundError:
        print(f"   HATA: '{dataset_path}' dosyası bulunamadı.")
        return

    # Temizleme
    mask = (
        (df.get('PoorAudioQuality', 0) == 0) & 
        (df.get('Music', 0) == 0) & 
        (df.get('NoSpeech', 0) == 0) & 
        (df.get('Unsure', 0) == 0)
    )
    df_clean = df[mask].copy()

    # Etiketleme
    print("2. Etiketleme Yapılıyor...")
    stutter_types = ['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection']
    
    def determine_label(row):
        for col in stutter_types:
            if col in row and row[col] > 0:
                return 1
        return 0

    df_clean['is_stutter'] = df_clean.apply(determine_label, axis=1)

    # Özellik Seçimi
    print("3. Özellikler Seçiliyor...")
    mfcc_features = [str(i) for i in range(13)]
    if 0 in df_clean.columns:
        mfcc_features = [i for i in range(13)]

    X = df_clean[mfcc_features].values
    y = df_clean['is_stutter'].values

    # Train/Test Split
    print("4. Train/Test Ayrımı ve Ölçeklendirme...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SCALING (Kritik Adım: Modeli kaydetmeden önce veriyi scale etmeliyiz)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    print("5. SMOTE ile Veri Dengeleniyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Modelleme
    print("6. Random Forest Modeli Eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Test
    print("7. Model Test Ediliyor...")
    y_pred = rf_model.predict(X_test_scaled)
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Fluent', 'Stutter']))

    # Kaydetme
    print("8. Model ve Scaler Kaydediliyor...")
    joblib.dump(rf_model, 'stutter_rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("   ✅ Model: stutter_rf_model.pkl")
    print("   ✅ Scaler: scaler.pkl")
    print("   İşlem Başarıyla Tamamlandı.")

if __name__ == "__main__":
    train_and_save_model()
