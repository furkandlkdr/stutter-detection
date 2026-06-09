"""Üç modelin eğitim süresini aynı koşullarda ölçer.

Karşılaştırma için gerçek saniye değerlerini üretir.
Sonuçlar JSON olarak kaydedilir; grafik scripti buradan okur.

Kullanım:
    .venv311/Scripts/python.exe scripts/benchmark_models.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "balanced_dataset.csv"
OUT = ROOT / "data" / "benchmark_results.json"

# balanced_dataset.csv: ilk 13 sütun = MFCC ortalamaları, 'is_stutter' = etiket.
# Bazı ara CSV'lerde 'mfcc_X_mean' adı kullanıldı; bu yüzden sütun adından bağımsız
# seçim yapıyoruz: son sütun hariç hepsi feature.
LABEL_COL = "is_stutter"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def _maybe_import_xgb():
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier
    except ImportError:
        return None


def _maybe_import_lgb():
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        return LGBMClassifier
    except None:
        return None


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {DATA}")
    df = pd.read_csv(DATA)
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"'{LABEL_COL}' sütunu bulunamadı. Mevcut: {df.columns.tolist()}"
        )
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    print(f"  Feature sütunları ({len(feature_cols)}): {feature_cols[:3]}... {feature_cols[-1]}")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.int32)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def main() -> None:
    print(f"Veri yükleniyor: {DATA}")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Eğitim: {X_train.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results: list[dict] = []

    # 1) Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.perf_counter()
    rf.fit(X_train_s, y_train)
    t_rf = time.perf_counter() - t0
    pred = rf.predict(X_test_s)
    results.append(
        {
            "model": "Random Forest",
            "train_time_s": round(t_rf, 2),
            "f1_stutter": round(f1_score(y_test, pred, pos_label=1), 4),
            "f1_fluent": round(f1_score(y_test, pred, pos_label=0), 4),
        }
    )
    print(f"  Random Forest : {t_rf:.2f} sn")

    # 2) XGBoost
    XGBClassifier = _maybe_import_xgb()
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
        t0 = time.perf_counter()
        xgb.fit(X_train_s, y_train)
        t_xgb = time.perf_counter() - t0
        pred = xgb.predict(X_test_s)
        results.append(
            {
                "model": "XGBoost",
                "train_time_s": round(t_xgb, 2),
                "f1_stutter": round(f1_score(y_test, pred, pos_label=1), 4),
                "f1_fluent": round(f1_score(y_test, pred, pos_label=0), 4),
            }
        )
        print(f"  XGBoost       : {t_xgb:.2f} sn")
    else:
        print("  XGBoost yok, atlanıyor (pip install xgboost)")

    # 3) LightGBM
    LGBMClassifier = _maybe_import_lgb()
    if LGBMClassifier is not None:
        lgb = LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        t0 = time.perf_counter()
        lgb.fit(X_train_s, y_train)
        t_lgb = time.perf_counter() - t0
        pred = lgb.predict(X_test_s)
        results.append(
            {
                "model": "LightGBM",
                "train_time_s": round(t_lgb, 2),
                "f1_stutter": round(f1_score(y_test, pred, pos_label=1), 4),
                "f1_fluent": round(f1_score(y_test, pred, pos_label=0), 4),
            }
        )
        print(f"  LightGBM      : {t_lgb:.2f} sn")
    else:
        print("  LightGBM yok, atlanıyor (pip install lightgbm)")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {OUT}")


if __name__ == "__main__":
    main()
