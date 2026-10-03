"""Görselleştirme yardımcıları.

Notebook'ta ve demo betiğinde kullanılır. Hepsi matplotlib Figure döndürür;
böylece çağıran taraf isterse kaydeder, isterse gösterir.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .metrics import LOWER_IS_BETTER, align, compute_metrics

# Yöntem başına sabit renk: tüm grafiklerde aynı yöntem aynı renkte görünsün.
METHOD_COLORS = {
    "otsu": "#4C72B0",
    "adaptive": "#DD8452",
    "kmeans": "#55A868",
    "watershed": "#C44E52",
}
FALLBACK_COLORS = ["#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def method_color(name: str, index: int = 0) -> str:
    return METHOD_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _hide_axes(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def show_stages(stages: dict, title: str | None = None):
    """Ön işleme zincirinin ara adımlarını yan yana gösterir."""
    order = [
        ("original", "Orijinal"),
        ("hair_mask", "Kıl maskesi"),
        ("dehaired", "Kıl temizlenmiş"),
        ("denoised", "Median filtre"),
        ("shade_corrected", "Gölge düzeltme"),
        ("enhanced", "CLAHE"),
    ]
    present = [(key, label) for key, label in order if key in stages]

    figure, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.6))
    if len(present) == 1:
        axes = [axes]

    for axis, (key, label) in zip(axes, present):
        image = stages[key]
        if image.ndim == 2:
            axis.imshow(image, cmap="gray")
        else:
            axis.imshow(image)
        axis.set_title(label, fontsize=10)
        _hide_axes(axis)

    if title:
        figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    return figure


def overlay_contours(
    image: np.ndarray,
    truth: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    truth_color: tuple[int, int, int] = (0, 220, 0),
    prediction_color: tuple[int, int, int] = (255, 40, 40),
    thickness: int = 2,
) -> np.ndarray:
    """Gerçek ve tahmin maskelerinin sınırlarını görüntü üzerine çizer.

    Maskeleri yarı saydam boyamak yerine kontur çizmek, sınırın nerede
    kaydığını görmeyi kolaylaştırır.
    """
    canvas = image.copy()
    for mask, color in ((truth, truth_color), (prediction, prediction_color)):
        if mask is None or not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, thickness)
    return canvas


def compare_methods(
    image: np.ndarray,
    truth: np.ndarray,
    pipelines: dict,
    metric: str = "dice",
):
    """Bir görüntüde tüm yöntemleri yan yana karşılaştırır.

    İlk sütun gerçek maske konturunu, sonraki sütunlar her yöntemin
    tahminini ve metrik değerini gösterir.
    """
    columns = len(pipelines) + 1
    figure, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 3.8))

    axes[0].imshow(overlay_contours(image, truth=truth))
    axes[0].set_title("Gerçek maske", fontsize=10)
    _hide_axes(axes[0])

    for axis, (name, pipeline) in zip(axes[1:], pipelines.items()):
        prediction = align(pipeline.predict(image), truth)
        score = compute_metrics(prediction, truth, with_boundary=False)[metric]
        axis.imshow(overlay_contours(image, truth=truth, prediction=prediction))
        axis.set_title(f"{name}\n{metric}={score:.3f}", fontsize=10)
        _hide_axes(axis)

    figure.tight_layout()
    return figure


def plot_metric_bars(summary: dict, metrics=("iou", "dice", "sensitivity", "precision")):
    """Yöntem x metrik gruplu bar grafiği.

    Args:
        summary: {yöntem: {metrik: değer}} sözlüğü.
    """
    methods = list(summary)
    metrics = [m for m in metrics if any(m in summary[k] for k in methods)]

    x = np.arange(len(methods), dtype=float)
    width = 0.8 / max(len(metrics), 1)

    figure, axis = plt.subplots(figsize=(1.6 * len(methods) + 4, 4.5))
    for index, metric in enumerate(metrics):
        values = [summary[m].get(metric, np.nan) for m in methods]
        offset = (index - (len(metrics) - 1) / 2) * width
        bars = axis.bar(x + offset, values, width, label=metric)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    fontsize=7,
                )

    axis.set_xticks(x)
    axis.set_xticklabels(methods)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Skor")
    axis.set_title("Yöntem karşılaştırması (yüksek = iyi)")
    axis.legend(ncol=len(metrics), fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def plot_distributions(results, metric: str = "dice"):
    """Metriğin yöntem başına dağılımını kutu grafiğiyle gösterir.

    Ortalama tek başına yanıltıcı olabilir: bir yöntem çoğu görüntüde iyi
    olup birkaçında tamamen başarısız olabilir. Dağılım bunu görünür kılar.
    """
    methods = sorted(results["method"].unique())
    data = [results.loc[results["method"] == m, metric].dropna().to_numpy() for m in methods]

    figure, axis = plt.subplots(figsize=(1.5 * len(methods) + 4, 4.5))
    # Etiketleri boxplot'a parametre olarak vermiyoruz: `labels` matplotlib
    # 3.9'da `tick_labels` olarak yeniden adlandırıldı, eksen üzerinden
    # ayarlamak her iki sürümde de çalışır.
    boxes = axis.boxplot(data, patch_artist=True, showmeans=True)
    axis.set_xticks(np.arange(1, len(methods) + 1))
    axis.set_xticklabels(methods)

    for index, (patch, name) in enumerate(zip(boxes["boxes"], methods)):
        patch.set_facecolor(method_color(name, index))
        patch.set_alpha(0.65)

    direction = "düşük = iyi" if metric in LOWER_IS_BETTER else "yüksek = iyi"
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} dağılımı ({direction}, n={len(data[0]) if data else 0})")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def plot_failure_cases(results, pipelines, dataset, metric: str = "dice", count: int = 3):
    """En kötü sonuç veren görüntüleri gösterir.

    Hangi görüntü tiplerinde başarısız olduğunu görmek, yöntemin sınırlarını
    anlamanın en hızlı yolu.
    """
    by_image = results.groupby("image")[metric].mean().sort_values()
    worst = list(by_image.index[:count])

    lookup = {sample.name: sample for sample in dataset}
    figures = []
    for name in worst:
        sample = lookup.get(name)
        if sample is None:
            continue
        image, truth = sample.load()
        figure = compare_methods(image, truth, pipelines, metric=metric)
        figure.suptitle(f"Zor örnek: {name} (ort. {metric}={by_image[name]:.3f})", fontsize=11)
        figures.append(figure)
    return figures
