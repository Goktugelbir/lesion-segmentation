"""Segmentasyon yöntemleri.

Her fonksiyon ön işlenmiş bir RGB görüntü ve geçerli bölge maskesi alır,
ham (son işleme yapılmamış) ikili maske döndürür. Son işleme ayrı modülde,
çünkü dört yöntem de aynı temizliği paylaşıyor — yöntemleri yalnızca eşik
kararı ayırt etsin istiyoruz.

Hepsinin ortak varsayımı: lezyon çevresindeki deriden daha koyudur.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_otsu


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _ones_like_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape[:2], dtype=np.uint8)


def segment_otsu(image: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """Global Otsu eşikleme.

    Eşik yalnızca geçerli bölgedeki piksellerden hesaplanır; siyah çerçeve
    histograma karışırsa eşik gereğinden fazla aşağı kayar.
    """
    fov = _ones_like_mask(image) if fov is None else fov
    gray = _gray(image)

    values = gray[fov > 0]
    if values.size == 0 or values.min() == values.max():
        return np.zeros_like(gray, dtype=np.uint8)

    threshold = threshold_otsu(values)
    return ((gray < threshold) & (fov > 0)).astype(np.uint8)


def segment_adaptive(
    image: np.ndarray,
    fov: np.ndarray | None = None,
    block_size: int | None = None,
    block_ratio: float = 0.25,
    constant: int = 5,
) -> np.ndarray:
    """Adaptif Gaussian eşikleme.

    Eşik her piksel için komşuluğundan hesaplanır, bu yüzden düzgün olmayan
    aydınlatmaya dayanıklıdır; karşılığında dokulu deride gürültülü maske
    üretir (son işleme bunu kısmen toparlıyor).

    `block_size` verilmezse görüntü boyutuna göre belirlenir. Bu kritik: blok
    lezyondan küçük kaldığında, lezyonun iç kısmında yerel ortalama piksel
    değerine eşitlenir ve hiçbir piksel "ortalamadan koyu" sayılmaz — yöntem
    yalnızca ince bir kenar halkası üretir. Blok lezyonu aşacak kadar büyük
    olduğunda yerel ortalamaya deri de karışır ve iç bölge de yakalanır.
    Sabit 51 piksellik blok, 600x450 dermoskopi görüntülerinde tam bu şekilde
    başarısız oluyordu.
    """
    fov = _ones_like_mask(image) if fov is None else fov
    gray = _gray(image)

    if block_size is None:
        block_size = int(block_ratio * min(gray.shape))

    # blockSize tek ve >1 olmak zorunda.
    block = max(3, block_size | 1)
    binary = cv2.adaptiveThreshold(
        gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, constant
    )
    return (binary & (fov > 0)).astype(np.uint8)


def segment_kmeans(
    image: np.ndarray,
    fov: np.ndarray | None = None,
    clusters: int = 3,
    spatial_weight: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """LAB uzayında k-means kümeleme; en koyu küme lezyon kabul edilir.

    Tek bir global eşik yerine renk uzayında gruplama yaptığı için, lezyonun
    deriyle parlaklık farkı az ama renk farkı belirgin olduğu durumlarda
    Otsu'dan iyi sonuç verir.

    `spatial_weight` > 0 verilirse piksel koordinatları da özniteliğe eklenir;
    bu kümeleri mekânsal olarak daha bitişik yapar ama lezyonun merkezde
    olduğu varsayımını güçlendirir.
    """
    fov = _ones_like_mask(image) if fov is None else fov
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    valid = fov > 0
    if valid.sum() < clusters:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    features = lab[valid].astype(np.float32)

    if spatial_weight > 0:
        height, width = image.shape[:2]
        ys, xs = np.nonzero(valid)
        # Koordinatları LAB ile benzer ölçeğe getir (0-255 aralığına yay).
        coords = np.stack(
            [ys / max(height - 1, 1) * 255.0, xs / max(width - 1, 1) * 255.0], axis=1
        ).astype(np.float32)
        features = np.hstack([features, coords * spatial_weight])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        features, clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # Merkezlerin L (parlaklık) bileşeni ilk sütunda; en küçüğü en koyu küme.
    darkest = int(np.argmin(centers[:, 0]))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[valid] = (labels.ravel() == darkest).astype(np.uint8)
    return mask


def segment_watershed(
    image: np.ndarray,
    fov: np.ndarray | None = None,
    foreground_ratio: float = 0.4,
) -> np.ndarray:
    """İşaretleyici tabanlı watershed.

    Otsu maskesinden kesin ön plan (uzaklık dönüşümünün tepesi) ve kesin arka
    plan işaretleyicileri çıkarılır; aradaki belirsiz bant watershed ile
    görüntünün gradyanına göre paylaştırılır. Bu, komşu lezyonların
    birleşmesini engeller ve sınırı gerçek kenara oturtur.
    """
    fov = _ones_like_mask(image) if fov is None else fov

    seed_mask = segment_otsu(image, fov)
    if not seed_mask.any():
        return seed_mask

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    if not opened.any():
        return seed_mask

    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    sure_fg = (distance > foreground_ratio * distance.max()).astype(np.uint8)
    if not sure_fg.any():
        return seed_mask

    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    unknown = sure_bg.astype(np.int16) - sure_fg.astype(np.int16)

    count, markers = cv2.connectedComponents(sure_fg)
    if count <= 1:
        return seed_mask

    # Arka plan 0'dan 1'e kayar, ön plan bileşenleri 2..n olur; belirsiz
    # bölgeyi 0 bırakıyoruz ki watershed oraya karar versin.
    markers = markers.astype(np.int32) + 1
    markers[unknown > 0] = 0

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(bgr, markers)

    # markers == -1 havza sınırı, == 1 arka plan, >= 2 nesneler.
    return ((markers >= 2) & (fov > 0)).astype(np.uint8)


# Yöntem adı -> fonksiyon. Değerlendirme betiği ve notebook bunu kullanır,
# böylece yeni bir yöntem eklemek tek satır oluyor.
SEGMENTERS = {
    "otsu": segment_otsu,
    "adaptive": segment_adaptive,
    "kmeans": segment_kmeans,
    "watershed": segment_watershed,
}
