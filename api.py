"""FastAPI tabanlı kekemelik tespit servisi.

Amaç: Eğitilmiş `stutter_rf_model.pkl` + `scaler.pkl` modelini dış dünyaya
HTTP üzerinden açmak. POST /analyze endpoint'i bir ses dosyası alır, 3 sn
chunk'lara böler, MFCC çıkarımı + scaler + model.predict_proba + threshold
(0.625) ile kekemelik tespiti yapar.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class UTF8JSONResponse(JSONResponse):
    """Türkçe karakterleri escape etmeden döndüren UTF-8 JSON yanıtı."""

    def render(self, content) -> bytes:
        import json

        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# --- Sabitler --------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "stutter_rf_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
SAMPLE_RATE = 16000
CHUNK_DURATION = 3
NOISE_GATE_RMS = 0.005
DECISION_THRESHOLD = 0.625


# --- Pydantic modelleri ----------------------------------------------------

class ChunkResult(BaseModel):
    start_time: float
    end_time: float
    label: str
    confidence: float


class AnalyzeResponse(BaseModel):
    total_duration: float
    stutter_count: int
    chunks: list[ChunkResult]


# --- Global durum ----------------------------------------------------------

class ModelState:
    model = None
    scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken model ve scaler'ı belleğe yükler."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Model/Scaler bulunamadı. Önce 'train_model.py' çalıştırın.\n"
            f"Beklenen: {MODEL_PATH} ve {SCALER_PATH}"
        )

    ModelState.model = joblib.load(MODEL_PATH)
    ModelState.scaler = joblib.load(SCALER_PATH)
    print(f"Model yüklendi: {MODEL_PATH.name} (threshold={DECISION_THRESHOLD})")

    yield

    ModelState.model = None
    ModelState.scaler = None


app = FastAPI(
    title="Stuttering Detection API",
    version="1.0.0",
    description="Ses dosyalarını 3 sn chunk'lara bölerek MFCC üzerinden "
    "kekemelik tespiti yapan servis.",
    default_response_class=UTF8JSONResponse,
    lifespan=lifespan,
)

# CORS: GitHub Pages / ngrok / her yerden erişim için en esnek ayar.
# NOT: allow_credentials=True + allow_origins=["*"] modern tarayıcılarda
# CORS spec'e aykırıdır ve preflight reddedilir. Bizim auth/cookie kullanımımız
# olmadığı için credentials=False kalıyor; böylece tüm origin'lere izin verilebilir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)


@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Tüm path'ler için açık OPTIONS preflight cevabı.

    CORSMiddleware genelde yeterli olur; ama ngrok ve bazı reverse-proxy
    konfigürasyonlarında preflight beklenmedik şekilde düşebiliyor. Bu
    handler her durumda 200 dönüp Access-Control-* başlıklarını ekler.
    """
    return UTF8JSONResponse(content={"ok": True})


# --- Yardımcılar -----------------------------------------------------------

def extract_mfcc_mean(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """(13,) MFCC ortalaması döndürür — eğitimdeki format."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def classify_chunk(
    chunk: np.ndarray,
    sample_rate: int,
    model,
    scaler,
    threshold: float,
) -> Optional[dict]:
    """Tek bir ses parçası için (label, confidence) döndürür. Sessizlik ise None."""
    # Çok kısa parçaları atla
    if len(chunk) < sample_rate:
        return None

    # Noise gate
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < NOISE_GATE_RMS:
        return {"label": "AKICI", "confidence": round(rms, 4), "is_stutter": 0}

    features = extract_mfcc_mean(chunk, sample_rate)
    features_scaled = scaler.transform(features.reshape(1, -1))

    probs = model.predict_proba(features_scaled)[0]
    stutter_prob = float(probs[1])
    prediction = 1 if stutter_prob > threshold else 0
    confidence = stutter_prob if prediction == 1 else float(probs[0])

    return {
        "label": "KEKEMELİK" if prediction == 1 else "AKICI",
        "confidence": round(confidence, 4),
        "is_stutter": prediction,
    }


# --- Endpoint'ler ----------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "Stuttering Detection API",
        "status": "ok",
        "threshold": DECISION_THRESHOLD,
        "chunk_duration": CHUNK_DURATION,
    }


@app.get("/health")
def health() -> dict:
    ready = ModelState.model is not None and ModelState.scaler is not None
    return {"ready": ready}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if ModelState.model is None or ModelState.scaler is None:
        raise HTTPException(status_code=503, detail="Model hazır değil.")

    # Geçici dosyaya kaydet
    raw_name = file.filename or "upload.wav"
    suffix = Path(raw_name).suffix or ".wav"
    temp_path = Path(tempfile.gettempdir()) / f"stutter_api_{uuid.uuid4().hex}{suffix}"

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")
        if len(contents) < 1024:
            raise HTTPException(
                status_code=400,
                detail=f"Ses dosyası çok küçük ({len(contents)} byte).",
            )
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Ses yükle (mono, hedef SR)
        try:
            y, sr = librosa.load(str(temp_path), sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Ses dosyası okunamadı: {exc}",
            )
        if len(y) == 0:
            raise HTTPException(status_code=400, detail="Ses verisi okunamadı.")

        total_duration = float(librosa.get_duration(y=y, sr=sr))
        samples_per_chunk = int(CHUNK_DURATION * sr)
        results: list[ChunkResult] = []
        stutter_count = 0
        # Waveform: tüm sinyalin küçük pencereler halinde peak değerleri.
        # Tarayıcı 0..1 arası yüksekliklere ihtiyaç duyar; ~60 çubuk yeterli.
        waveform_bars = 60
        waveform_step = max(1, len(y) // waveform_bars)
        waveform = [
            float(min(1.0, max(0.0, abs(float(y[j])))))
            for j in range(0, len(y), waveform_step)
        ][:waveform_bars]

        for i in range(0, len(y), samples_per_chunk):
            chunk = y[i:i + samples_per_chunk]
            decision = classify_chunk(
                chunk=chunk,
                sample_rate=sr,
                model=ModelState.model,
                scaler=ModelState.scaler,
                threshold=DECISION_THRESHOLD,
            )
            if decision is None:
                continue

            start_time = round(i / sr, 2)
            end_time = round((i + len(chunk)) / sr, 2)
            results.append(
                ChunkResult(
                    start_time=start_time,
                    end_time=end_time,
                    label=decision["label"],
                    confidence=decision["confidence"],
                )
            )
            if decision["is_stutter"] == 1:
                stutter_count += 1

        return UTF8JSONResponse(
            content={
                "total_duration": round(total_duration, 2),
                "stutter_count": stutter_count,
                "waveform": waveform,
                "chunks": [
                    {
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "label": chunk.label,
                        "confidence": chunk.confidence,
                    }
                    for chunk in results
                ],
            }
        )

    finally:
        # Geçici dosyayı her durumda temizle
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except OSError:
            pass


# --- Çalıştırma bloğu ------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
