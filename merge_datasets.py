from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
SEP28K_PATH = ROOT_DIR / "sep28k-mfcc.csv"
FLUENT_PATH = ROOT_DIR / "fluent_mfcc.csv"
OUTPUT_PATH = ROOT_DIR / "balanced_dataset.csv"

MFCC_COLUMNS = [str(i) for i in range(13)]
STUTTER_COLUMNS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
CLEAN_COLUMNS = ["PoorAudioQuality", "Music", "NoSpeech", "Unsure"]


def build_sep28k_frame() -> pd.DataFrame:
    if not SEP28K_PATH.exists():
        raise FileNotFoundError(f"SEP-28k dosyası bulunamadı: {SEP28K_PATH}")

    df = pd.read_csv(SEP28K_PATH)

    for column in CLEAN_COLUMNS:
        if column not in df.columns:
            raise KeyError(f"Eksik sütun: {column}")

    clean_mask = (
        (df["PoorAudioQuality"] == 0)
        & (df["Music"] == 0)
        & (df["NoSpeech"] == 0)
        & (df["Unsure"] == 0)
    )
    df = df[clean_mask].copy()

    missing_mfcc = [column for column in MFCC_COLUMNS if column not in df.columns]
    if missing_mfcc:
        raise KeyError(f"SEP-28k içinde eksik MFCC sütunları: {missing_mfcc}")

    def determine_label(row: pd.Series) -> int:
        for column in STUTTER_COLUMNS:
            if column in row and row[column] > 0:
                return 1
        return 0

    df["is_stutter"] = df.apply(determine_label, axis=1)
    return df[MFCC_COLUMNS + ["is_stutter"]].copy()


def build_fluent_frame() -> pd.DataFrame:
    if not FLUENT_PATH.exists():
        raise FileNotFoundError(f"Fluent dosyası bulunamadı: {FLUENT_PATH}")

    df = pd.read_csv(FLUENT_PATH)

    missing_columns = [column for column in MFCC_COLUMNS + ["is_stutter"] if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Fluent dosyasında eksik sütunlar: {missing_columns}")

    return df[MFCC_COLUMNS + ["is_stutter"]].copy()


def main() -> None:
    sep28k_frame = build_sep28k_frame()
    fluent_frame = build_fluent_frame()

    balanced_frame = pd.concat([sep28k_frame, fluent_frame], ignore_index=True)
    balanced_frame = balanced_frame.sample(frac=1, random_state=42).reset_index(drop=True)

    balanced_frame.to_csv(OUTPUT_PATH, index=False)

    print(f"Kaydedildi: {OUTPUT_PATH}")
    print("is_stutter dağılımı:")
    print(balanced_frame["is_stutter"].value_counts().sort_index())


if __name__ == "__main__":
    main()