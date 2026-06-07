import pandas as pd
import joblib
from pathlib import Path
import sys
import types


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "balanced_dataset.csv"


def install_scipy_propack_shim() -> None:
    shim = types.ModuleType("scipy.sparse.linalg._propack")

    def _stub(*args, **kwargs):
        raise NotImplementedError("SciPy PROPACK shim is only for import compatibility.")

    class _PropackNamespace:
        def __getattr__(self, name):
            return _stub

    propack_namespace = _PropackNamespace()

    def _module_getattr(name: str):
        if name in {"_spropack", "_dpropack", "_cpropack", "_zpropack"}:
            return propack_namespace
        return _stub

    shim.__getattr__ = _module_getattr
    sys.modules["scipy.sparse.linalg._propack"] = shim

def train_and_save_model():
    install_scipy_propack_shim()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("1. Veri Yükleniyor...")
    if not DATASET_PATH.exists():
        print(f"   HATA: '{DATASET_PATH}' dosyası bulunamadı.")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"   Veri seti yüklendi: {DATASET_PATH}")

    print("2. Özellikler Seçiliyor...")
    mfcc_features = [str(i) for i in range(13)]
    X = df[mfcc_features].values
    y = df['is_stutter'].values

    # Train/Test Split
    print("3. Train/Test Ayrımı ve Ölçeklendirme...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SCALING (Kritik Adım: Modeli kaydetmeden önce veriyi scale etmeliyiz)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelleme
    print("4. Random Forest Modeli Eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    # Test
    print("5. Model Test Ediliyor...")
    y_pred = rf_model.predict(X_test_scaled)
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Fluent', 'Stutter']))

    # Kaydetme
    print("6. Model ve Scaler Kaydediliyor...")
    joblib.dump(rf_model, 'stutter_rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("   ✅ Model: stutter_rf_model.pkl")
    print("   ✅ Scaler: scaler.pkl")
    print("   İşlem Başarıyla Tamamlandı.")

if __name__ == "__main__":
    train_and_save_model()
