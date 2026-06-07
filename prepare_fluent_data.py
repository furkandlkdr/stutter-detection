from __future__ import annotations

import argparse
import csv
import math
import wave
import random
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = [str(i) for i in range(13)] + ["is_stutter"]

N_MFCC = 13
N_MELS = 26
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
PRE_EMPHASIS = 0.97


def hz_to_mel(freq_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def read_wav_mono(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        raw_audio = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Desteklenmeyen sample width: {sample_width * 8} bit")

    audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1)

    audio /= 32768.0

    if sample_rate != target_sample_rate and len(audio) > 1:
        duration_seconds = len(audio) / sample_rate
        old_times = np.linspace(0.0, duration_seconds, num=len(audio), endpoint=False)
        new_length = int(round(duration_seconds * target_sample_rate))
        new_times = np.linspace(0.0, duration_seconds, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
        sample_rate = target_sample_rate

    return audio, sample_rate


def frame_audio(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))

    frame_count = 1 + max(0, (len(audio) - frame_length) // hop_length)
    frames = np.empty((frame_count, frame_length), dtype=np.float32)

    for index in range(frame_count):
        start = index * hop_length
        frames[index] = audio[start:start + frame_length]

    return frames


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for mel_index in range(1, n_mels + 1):
        left = bin_points[mel_index - 1]
        center = bin_points[mel_index]
        right = bin_points[mel_index + 1]

        if center == left:
            center += 1
        if right == center:
            right += 1

        for fft_index in range(left, center):
            if 0 <= fft_index < filterbank.shape[1]:
                filterbank[mel_index - 1, fft_index] = (fft_index - left) / (center - left)
        for fft_index in range(center, right):
            if 0 <= fft_index < filterbank.shape[1]:
                filterbank[mel_index - 1, fft_index] = (right - fft_index) / (right - center)

    return filterbank


def extract_mfcc_mean(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    emphasized = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])
    frames = frame_audio(emphasized, WIN_LENGTH, HOP_LENGTH)
    frames *= np.hamming(WIN_LENGTH).astype(np.float32)

    power_spectrum = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    power_spectrum /= N_FFT

    filterbank = build_mel_filterbank(sample_rate, N_FFT, N_MELS)
    mel_energies = np.dot(power_spectrum, filterbank.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    log_mel_energies = np.log(mel_energies)

    n_filters = log_mel_energies.shape[1]
    dct_basis = np.cos(
        np.pi
        / n_filters
        * (np.arange(N_MFCC)[:, None])
        * (np.arange(n_filters)[None, :] + 0.5)
    )
    dct_basis[0] *= 1.0 / np.sqrt(2.0)
    mfcc = np.dot(log_mel_energies, dct_basis.T)
    mfcc *= np.sqrt(2.0 / n_filters)

    return mfcc.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fluent MFCC samples from VoxCeleb WAV files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "voxceleb" / "vox1_dev_wav",
        help="Root directory that contains VoxCeleb WAV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fluent_mfcc.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=3500,
        help="Number of fluent samples to generate.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Random chunk duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target audio sample rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def find_wav_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {root_dir}")

    wav_files = [path for path in root_dir.rglob("*.wav") if path.is_file()]
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {root_dir}")

    return wav_files


def extract_mfcc_mean(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    emphasized = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])
    frames = frame_audio(emphasized, WIN_LENGTH, HOP_LENGTH)
    frames *= np.hamming(WIN_LENGTH).astype(np.float32)

    power_spectrum = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    power_spectrum /= N_FFT

    filterbank = build_mel_filterbank(sample_rate, N_FFT, N_MELS)
    mel_energies = np.dot(power_spectrum, filterbank.T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    log_mel_energies = np.log(mel_energies)

    n_filters = log_mel_energies.shape[1]
    dct_basis = np.cos(
        np.pi
        / n_filters
        * (np.arange(N_MFCC)[:, None])
        * (np.arange(n_filters)[None, :] + 0.5)
    )
    dct_basis[0] *= 1.0 / np.sqrt(2.0)
    mfcc = np.dot(log_mel_energies, dct_basis.T)
    mfcc *= np.sqrt(2.0 / n_filters)

    return mfcc.mean(axis=0)


def generate_fluent_samples(
    wav_files: list[Path],
    target_samples: int,
    chunk_duration: float,
    sample_rate: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    shuffled_files = wav_files[:]
    rng.shuffle(shuffled_files)

    chunk_samples = int(chunk_duration * sample_rate)
    rows: list[list[float | int]] = []
    scanned_files = 0
    skipped_short = 0

    print(f"Toplam aday dosya: {len(shuffled_files)}")
    print(f"Hedef örnek sayısı: {target_samples}")
    print(f"Chunk süresi: {chunk_duration:.1f} saniye")
    print("İşlem başlıyor...")

    while len(rows) < target_samples:
        if scanned_files >= len(shuffled_files):
            shuffled_files = wav_files[:]
            rng.shuffle(shuffled_files)
            scanned_files = 0

        wav_path = shuffled_files[scanned_files]
        scanned_files += 1

        try:
            audio, _ = read_wav_mono(wav_path, sample_rate)
        except Exception as exc:
            print(f"Atlandı (okunamadı): {wav_path} | {exc}")
            continue

        if len(audio) < chunk_samples:
            skipped_short += 1
            continue

        max_start = len(audio) - chunk_samples
        start_sample = rng.randint(0, max_start) if max_start > 0 else 0
        chunk = audio[start_sample:start_sample + chunk_samples]

        if len(chunk) < chunk_samples:
            skipped_short += 1
            continue

        mfcc_mean = extract_mfcc_mean(chunk, sample_rate)
        row = mfcc_mean.tolist() + [0]
        rows.append(row)

        if len(rows) % 100 == 0 or len(rows) == target_samples:
            print(
                f"İlerleme: {len(rows)}/{target_samples} örnek tamamlandı | "
                f"Taranan dosya: {scanned_files} | Kısa atlanan: {skipped_short}"
            )

    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_path = args.output.resolve()

    wav_files = find_wav_files(source_dir)
    df = generate_fluent_samples(
        wav_files=wav_files,
        target_samples=args.target_samples,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print("Tamamlandı.")
    print(f"Kaydedildi: {output_path}")
    print(f"Oluşturulan örnek sayısı: {len(df)}")


if __name__ == "__main__":
    main()