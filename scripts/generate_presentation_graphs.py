"""Bitirme projesi sunumu için grafik üretici.

Üç adet yüksek çözünürlüklü (300 DPI) PNG üretir:
    1. data_evolution.png        — Sınıf dağılımının evrimi (3 aşama)
    2. model_comparison.png      — Random Forest / XGBoost / LightGBM karşılaştırması
    3. threshold_impact.png      — Eşik optimizasyonunun etkisi (önce/sonra)

Renk paleti: modern ve akademik — mavi, gri, turuncu.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Sabitler ve ortak stil -------------------------------------------------

COLOR_BLUE = "#1F4E79"
COLOR_GREY = "#7F7F7F"
COLOR_GREY_LIGHT = "#D9D9D9"
COLOR_ORANGE = "#ED7D31"
COLOR_ORANGE_DARK = "#C55A11"
COLOR_BG = "#FFFFFF"

DPI = 300
FIGSIZE_LANDSCAPE = (11, 5.5)
FIGSIZE_PANEL = (10, 6)

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
BENCHMARK_PATH = ROOT_DIR / "data" / "benchmark_results.json"

# Artık eğitim süresi göstermiyoruz (inference'dan bağımsız, seçim kriteri değil).
# Grafik sınıf-bazlı F1 skorlarına odaklanıyor.

# Proje tablosundan (raporlanmış) — sınıf-bazlı F1 değerleri sunumda
# model seçim gerekçesini daha iyi anlatır.
DEFAULT_BENCHMARK = {
    "Random Forest": {"f1_stutter": 0.7739, "f1_fluent": 0.6656, "macro_f1": 0.7198},
    "XGBoost":        {"f1_stutter": 0.7339, "f1_fluent": 0.6704, "macro_f1": 0.7022},
    "LightGBM":       {"f1_stutter": 0.7335, "f1_fluent": 0.6718, "macro_f1": 0.7027},
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    }
)


def ensure_figures_dir() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _format_int(v: float) -> str:
    return f"{int(v):,}".replace(",", ".")


# --- 1. Veri Seti Evrimi ----------------------------------------------------

def graph_data_evolution() -> None:
    stages = [
        "1. Aşama\nSadece SEP-28k",
        "2. Aşama\nDengesiz Birleşim",
        "3. Aşama\nUndersampling",
    ]
    # balanced_dataset.csv: 7.905 fluent / 16.305 stutter (1:1 dengesizliği)
    stutter_counts = [16_305, 16_305, 7_905]
    fluent_counts = [0, 7_905, 7_905]

    fig, ax = plt.subplots(figsize=FIGSIZE_LANDSCAPE)

    x = np.arange(len(stages))
    width = 0.35

    bars_stutter = ax.bar(
        x - width / 2,
        stutter_counts,
        width,
        label="Kekemelik (Stutter)",
        color=COLOR_ORANGE,
        edgecolor=COLOR_ORANGE_DARK,
        linewidth=0.8,
    )
    bars_fluent = ax.bar(
        x + width / 2,
        fluent_counts,
        width,
        label="Akıcı (Fluent)",
        color=COLOR_BLUE,
        edgecolor="#14365A",
        linewidth=0.8,
    )

    for bar in list(bars_stutter) + list(bars_fluent):
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.01,
            _format_int(height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Örnek Sayısı", fontsize=11)
    ax.set_title("Veri Seti Evrimi — Organik Dengeleme Stratejisi", pad=14)
    ax.set_ylim(0, max(stutter_counts) * 1.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _format_int(v)))
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=COLOR_GREY_LIGHT)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC")

    fig.text(
        0.5,
        0.005,
        "1:1 denge, sentetik yöntem olmadan gerçek VoxCeleb verisi ile sağlandı.",
        ha="center",
        va="bottom",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    output = FIGURES_DIR / "data_evolution.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {output.name}")


# --- 2. Model Karşılaştırması ----------------------------------------------

def _load_benchmark() -> dict:
    """benchmark_models.py çıktısını okur; yoksa varsayılan değerleri kullanır.

    Not: Grafik artık sınıf-bazlı F1 skorlarını gösteriyor. JSON'dan
    yalnızca eğitim süreleri geliyor; sınıf F1'leri DEFAULT'tan okunur.
    """
    return DEFAULT_BENCHMARK


def graph_model_comparison() -> None:
    """Üç modelin sınıf-bazlı F1 skorlarını yan yana karşılaştırır.

    Eğitim süresi bilinçli olarak gösterilmez — inference süresinden bağımsız
    olduğu için model seçim kriteri değil. Seçim gerekçesi: en yüksek
    Macro F1 + en yüksek Stutter F1 (bizim asıl hedef sınıfımız).
    """
    bench = _load_benchmark()
    order = ["Random Forest", "XGBoost", "LightGBM"]
    models = [m for m in order if m in bench]

    f1_stutter = [bench[m]["f1_stutter"] for m in models]
    f1_fluent = [bench[m]["f1_fluent"] for m in models]
    macro_f1  = [bench[m]["macro_f1"]  for m in models]

    # Seçilen model = Random Forest (en yüksek Macro F1 + en yüksek Stutter F1)
    chosen_name = "Random Forest"
    chosen_idx = models.index(chosen_name) if chosen_name in models else 0

    fig, ax = plt.subplots(figsize=FIGSIZE_LANDSCAPE)

    x = np.arange(len(models))
    width = 0.27

    bars_st = ax.bar(
        x - width, f1_stutter, width,
        label="Stutter F1", color=COLOR_ORANGE,
        edgecolor=COLOR_ORANGE_DARK, linewidth=0.8,
    )
    bars_fl = ax.bar(
        x, f1_fluent, width,
        label="Fluent F1", color=COLOR_BLUE,
        edgecolor="#14365A", linewidth=0.8,
    )
    bars_ma = ax.bar(
        x + width, macro_f1, width,
        label="Macro F1", color=COLOR_GREY,
        edgecolor="#5A5A5A", linewidth=0.8,
    )

    for bar in list(bars_st) + list(bars_fl) + list(bars_ma):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.008,
            f"{h:.3f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#222222",
        )

    y_max = max(max(f1_stutter), max(f1_fluent), max(macro_f1))
    ax.set_ylim(0.55, y_max + 0.10)
    ax.set_ylabel("F1 Skoru", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # Seçilen model vurgusu
    ax.annotate(
        "SEÇİLEN MODEL",
        xy=(chosen_idx, f1_stutter[chosen_idx]),
        xytext=(chosen_idx, y_max + 0.06),
        ha="center", fontsize=10, fontweight="bold",
        color=COLOR_ORANGE_DARK,
        arrowprops=dict(arrowstyle="->", color=COLOR_ORANGE_DARK, lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_title("Model Karşılaştırması — Sınıf Bazlı F1 Skorları", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=COLOR_GREY_LIGHT)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC")

    fig.text(
        0.5, 0.005,
        "Random Forest hem Macro F1'de hem Stutter F1'de en yüksek. ",
        ha="center", va="bottom", fontsize=10, style="italic", color="#555555",
    )

    output = FIGURES_DIR / "model_comparison.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {output.name}")
# --- 3. Threshold Optimizasyonu --------------------------------------------

def graph_threshold_impact() -> None:
    scenarios = [
        "Eşik = 0.50\n(varsayılan)",
        "Eşik = 0.625\n(optimize)",
    ]
    fluent_recall = [54.0, 67.8]
    false_positive = [0.08, 0.04]

    fig, ax1 = plt.subplots(figsize=FIGSIZE_PANEL)

    x = np.arange(len(scenarios))
    width = 0.35

    bars_recall = ax1.bar(
        x - width / 2,
        fluent_recall,
        width,
        label="Fluent Recall (%)",
        color=COLOR_ORANGE,
        edgecolor=COLOR_ORANGE_DARK,
        linewidth=0.8,
    )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Fluent Recall (%)", fontsize=11, color=COLOR_ORANGE_DARK)
    ax1.tick_params(axis="y", labelcolor=COLOR_ORANGE_DARK)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))

    for bar in bars_recall:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"%{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=COLOR_ORANGE_DARK,
        )

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    bars_fp = ax2.bar(
        x + width / 2,
        [v * 100 for v in false_positive],
        width,
        label="False Positive Oranı (%)",
        color=COLOR_GREY,
        edgecolor="#5A5A5A",
        linewidth=0.8,
    )
    ax2.set_ylabel("False Positive Oranı (%)", fontsize=11, color="#5A5A5A")
    ax2.tick_params(axis="y", labelcolor="#5A5A5A")
    ax2.set_ylim(0, 12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"%{v:.1f}"))

    for bar in bars_fp:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"%{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#5A5A5A",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=11)
    ax1.set_title(
        "Eşik (Threshold) Optimizasyonu — Fluent Recall ve False Positive",
        pad=14,
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color=COLOR_GREY_LIGHT)
    ax1.set_axisbelow(True)

    ax1.annotate(
        "",
        xy=(1 - width / 2, fluent_recall[1]),
        xytext=(0 - width / 2, fluent_recall[0]),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.8),
    )
    ax1.text(
        0.5,
        max(fluent_recall) + 12,
        f"+{fluent_recall[1] - fluent_recall[0]:.1f} puan",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLOR_BLUE,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
    )

    fig.text(
        0.5,
        0.005,
        "F1 eğrisinden optimize edilen eşik; akıcı konuşmaları doğru tanıma oranı %13.8 arttı.",
        ha="center",
        va="bottom",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    output = FIGURES_DIR / "threshold_impact.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {output.name}")


def graph_undersampling_impact() -> None:
    """Undersampling öncesi (dengesiz) ve sonrası (1:1) metrik karşılaştırması.

    Değerler eğitim/değerlendirme çıktısından alındı (aynı test bölmesi).
    """
    metrics = ["Accuracy", "Fluent\nPrecision", "Fluent\nRecall", "Fluent F1",
               "Stutter\nPrecision", "Stutter\nRecall", "Stutter F1"]
    before = [0.8176, 1.00, 0.44, 0.61, 0.79, 1.00, 0.88]
    after  = [0.7302, 0.88, 0.54, 0.67, 0.67, 0.92, 0.77]

    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)

    x = np.arange(len(metrics))
    width = 0.35

    bars_before = ax.bar(
        x - width / 2, before, width,
        label="Önce (dengesiz 1:2.06)", color=COLOR_GREY,
        edgecolor="#5A5A5A", linewidth=0.8,
    )
    bars_after = ax.bar(
        x + width / 2, after, width,
        label="Sonra (1:1 dengeli)", color=COLOR_ORANGE,
        edgecolor=COLOR_ORANGE_DARK, linewidth=0.8,
    )

    for bar in list(bars_before) + list(bars_after):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.018,
            f"{h:.2f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222222",
        )

    # Fluent Recall: en kritik metrik — iyileşti
    fluent_recall_idx = metrics.index("Fluent\nRecall")
    ax.annotate(
        "+0.10",
        xy=(fluent_recall_idx + width / 2, after[fluent_recall_idx]),
        xytext=(fluent_recall_idx + width / 2 + 0.4, after[fluent_recall_idx] + 0.18),
        fontsize=11, fontweight="bold", color=COLOR_BLUE,
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Skor (0–1)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title("Undersampling Etkisi — Sınıflandırma Metrikleri (Önce / Sonra)", pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=COLOR_GREY_LIGHT)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC")

    fig.text(
        0.5, 0.005,
        "1:1 denge, gerçek akıcı veri ile sağlandı — Fluent Recall %44 → %54, F1 %61 → %67.",
        ha="center", va="bottom", fontsize=10, style="italic", color="#555555",
    )

    output = FIGURES_DIR / "undersampling_impact.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {output.name}")


def main() -> None:
    ensure_figures_dir()
    print(f"Grafikler üretiliyor: {FIGURES_DIR}")
    graph_data_evolution()
    graph_model_comparison()
    graph_threshold_impact()
    graph_undersampling_impact()
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
