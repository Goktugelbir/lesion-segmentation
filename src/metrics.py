"""Segmentasyon başarı metrikleri.

Bölge örtüşmesi (IoU, Dice) tek başına yetersiz: iki maske aynı Dice'ı verip
çok farklı sınır kalitesine sahip olabilir. Bu yüzden sınır mesafesi
metrikleri (HD95, ASSD) de hesaplanıyor.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# Değerlendirme çıktılarında kullanılan sabit kolon sırası.
METRIC_NAMES = (
    "iou",
    "dice",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "xor_error",
    "hd95",
    "assd",
)

# Bu metriklerde küçük değer daha iyidir (sıralama ve grafikler için gerekli).
LOWER_IS_BETTER = frozenset({"xor_error", "hd95", "assd"})


def align(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Maskeyi referans maskenin boyutuna getirir.

    İkili maske olduğu için en yakın komşu kullanılır; ara değer üretmek
    0/1 dışında piksel oluşturur.
    """
    if mask.shape == reference.shape:
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


def confusion(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """TP/FP/FN/TN sayılarını döndürür."""
    predicted = prediction.astype(bool)
    actual = truth.astype(bool)
    return {
        "tp": int(np.count_nonzero(predicted & actual)),
        "fp": int(np.count_nonzero(predicted & ~actual)),
        "fn": int(np.count_nonzero(~predicted & actual)),
        "tn": int(np.count_nonzero(~predicted & ~actual)),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    """Payda sıfırsa NaN döndürür.

    0.0 döndürmek yanıltıcı olurdu: "tanım gereği hesaplanamaz" ile "ölçüldü
    ve sıfır çıktı" aynı şey değil. NaN ortalamalardan dışlanabilir.
    """
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Maskenin sınır piksellerini döndürür."""
    mask = mask.astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask - eroded).astype(bool)


def boundary_distances(
    prediction: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Her iki yöndeki sınır-sınır mesafelerini hesaplar (piksel cinsinden).

    Returns:
        (tahmin sınırından gerçeğe mesafeler, gerçek sınırdan tahmine mesafeler)
        Maskelerden biri boşsa iki boş dizi döner.
    """
    predicted_edge = _boundary(prediction)
    truth_edge = _boundary(truth)

    empty = np.array([], dtype=np.float64)
    if not predicted_edge.any() or not truth_edge.any():
        return empty, empty

    # distance_transform_edt sıfır olmayan değerlerden uzaklığı değil, sıfır
    # olanlara uzaklığı ölçer; bu yüzden sınırı tersine çevirip veriyoruz.
    distance_to_truth = distance_transform_edt(~truth_edge)
    distance_to_prediction = distance_transform_edt(~predicted_edge)

    return (
        distance_to_truth[predicted_edge].astype(np.float64),
        distance_to_prediction[truth_edge].astype(np.float64),
    )


def hausdorff95(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Simetrik sınır mesafelerinin 95. yüzdeliği.

    Klasik Hausdorff tek bir aykırı piksele aşırı duyarlı olduğu için
    tıbbi görüntülemede 95. yüzdelik tercih edilir.
    """
    forward, backward = boundary_distances(prediction, truth)
    if forward.size == 0 or backward.size == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def average_surface_distance(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Ortalama simetrik yüzey mesafesi (ASSD)."""
    forward, backward = boundary_distances(prediction, truth)
    if forward.size == 0 or backward.size == 0:
        return float("nan")
    return float(np.concatenate([forward, backward]).mean())


def compute_metrics(
    prediction: np.ndarray, truth: np.ndarray, with_boundary: bool = True
) -> dict[str, float]:
    """Tüm metrikleri tek sözlükte döndürür.

    Args:
        prediction: 0/1 tahmin maskesi.
        truth: 0/1 referans maskesi (aynı boyutta olmalı; `align` kullanın).
        with_boundary: HD95/ASSD hesaplansın mı. Mesafe dönüşümü pahalı
            olduğu için büyük taramalarda kapatılabilir.
    """
    if prediction.shape != truth.shape:
        raise ValueError(
            f"Maske boyutlari uyusmuyor: {prediction.shape} != {truth.shape}"
        )

    counts = confusion(prediction, truth)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total = tp + fp + fn + tn

    results = {
        "iou": _safe_divide(tp, tp + fp + fn),
        "dice": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "accuracy": _safe_divide(tp + tn, total),
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "precision": _safe_divide(tp, tp + fp),
        # XOR hata oranı: ISIC değerlendirmelerinde kullanılan, lezyon alanına
        # normalize edilmiş hata. Küçük lezyonlarda accuracy'den çok daha ayırt edici.
        "xor_error": _safe_divide(fp + fn, tp + fn),
    }

    if with_boundary:
        results["hd95"] = hausdorff95(prediction, truth)
        results["assd"] = average_surface_distance(prediction, truth)

    return results
