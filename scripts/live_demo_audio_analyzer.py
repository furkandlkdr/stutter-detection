"""Yerel mikrofon ile canlı kekemelik tespiti (CLI demo).

Aynı iş mantığı (3 sn'lik chunklar, MFCC, scaler, threshold=0.625) FastAPI
üzerinden api.py'da da çalışır. Bu script, ekibin websitesiz komut satırından
hızlıca doğrulama yapabilmesi içindir.

Kullanım:
    python scripts/live_demo_audio_analyzer.py
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

# Librosa uyarılarını gizle (FP enforcer vs.)
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "stutter_rf_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Eğitimde kullanılan sabitlerle aynı
SAMPLE_RATE = 16_000
CHUNK_DURATION = 3
NOISE_GATE_RMS = 0.005
DECISION_THRESHOLD = 0.625
OUTPUT_FILENAME = "test_session.wav"


def record_audio(duration: int = 10, filename: str = OUTPUT_FILENAME) -> str:
    """Mikrofondan `duration` saniye ses kaydeder, 16 kHz mono WAV olarak yazar."""
    import pyaudio
    import wave

    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    print(f"\n🎙️  Kayıt Başlıyor... ({duration} saniye)")
    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=chunk,
    )

    frames: list[bytes] = []
    total_chunks = int(SAMPLE_RATE / chunk * duration)
    for i in range(total_chunks):
        frames.append(stream.read(chunk))
        if i % int(SAMPLE_RATE / chunk) == 0:
            remaining = duration - int(i / (SAMPLE_RATE / chunk))
            print(f"   Kalan Süre: {remaining} sn ", end="\r")

    print("\n✅ Kayıt Tamamlandı.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    print(f"   Dosya kaydedildi: {filename}")
    return filename


def extract_mfcc_mean(y: np.ndarray, sr: int) -> np.ndarray:
    """(13,) MFCC ortalaması döndürür — eğitimdeki format."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def analyze_file(
    file_path: str,
    model,
    scaler,
    threshold: float = DECISION_THRESHOLD,
    chunk_duration: int = CHUNK_DURATION,
) -> pd.DataFrame:
    """Ses dosyasını parçalara böler ve her parça için kekemelik tahmini yapar."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_duration = librosa.get_duration(y=y, sr=sr)
    print(f"   Toplam Süre: {total_duration:.2f} saniye")

    results: list[dict] = []
    samples_per_chunk = int(chunk_duration * sr)

    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i : i + samples_per_chunk]
        if len(chunk) < sr:
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < NOISE_GATE_RMS:
            start_time = i / sr
            end_time = (i + len(chunk)) / sr
            results.append(
                {
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "is_stutter": 0,
                    "confidence": 1.0,
                    "label": "SESSİZ/AKICI",
                }
            )
            continue

        features = extract_mfcc_mean(chunk, sr)
        features_scaled = scaler.transform(features.reshape(1, -1))
        probs = model.predict_proba(features_scaled)[0]

        # Threshold üzerindeyse KEKEMELİK
        is_stutter = probs[1] > threshold
        confidence = float(probs[1] if is_stutter else probs[0])

        results.append(
            {
                "start_time": round(i / sr, 2),
                "end_time": round((i + len(chunk)) / sr, 2),
                "is_stutter": int(is_stutter),
                "confidence": round(confidence, 2),
                "label": "KEKEMELİK" if is_stutter else "AKICI",
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Model/Scaler bulunamadı. Önce `python scripts/train_model.py` çalıştırın.\n"
            f"Beklenen: {MODEL_PATH} ve {SCALER_PATH}"
        )

    print("🧠 Model ve Scaler Yükleniyor...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"   Model yüklendi (threshold={DECISION_THRESHOLD}).")

    print("\n🎙️  Mikrofon Kaydı")
    wav_path = record_audio(duration=10, filename=OUTPUT_FILENAME)

    print("\n🔍 Analiz Ediliyor:", wav_path)
    df = analyze_file(wav_path, model, scaler)

    print("\n📊 ANALİZ SONUÇLARI (Zaman Çizelgesi)")
    print("=" * 60)
    print(f"{'Zaman Aralığı':<20} | {'Durum':<15} | {'Güven':<10}")
    print("-" * 60)
    stutter_count = 0
    for _, row in df.iterrows():
        time_str = f"{row['start_time']}s - {row['end_time']}s"
        prefix = "🔴 " if row["is_stutter"] == 1 else "🟢 "
        conf = f"%{row['confidence'] * 100:.1f}"
        print(f"{prefix} {time_str:<17} | {row['label']:<15} | {conf:<10}")
        if row["is_stutter"] == 1:
            stutter_count += 1
    print("=" * 60)
    print(f"Toplam Parça: {len(df)}")
    print(f"Tespit Edilen Kekemelik Sayısı: {stutter_count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")
        print(
            "Gerekli kütüphanelerin yüklü olduğundan emin olun:\n"
            "  pip install pyaudio librosa joblib pandas scikit-learn"
        )
