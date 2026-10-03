"""Ham segmentasyon maskesini kullanılabilir hale getiren son işleme.

Dört yöntem de aynı zinciri kullanır: gürültü temizliği, delik doldurma,
kenar yumuşatma ve tek bir lezyon bölgesinin seçilmesi.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


@dataclass
class PostprocessConfig:
    """Son işleme parametreleri.

    `min_area_ratio` görüntü alanına oranla verilir; böylece farklı
    çözünürlükteki görüntülerde aynı davranış korunur.
    """

    min_area_ratio: float = 0.002
    fill_holes: bool = True
    closing_size: int = 15
    opening_size: int = 5
    select: str = "center"  # "center" | "largest" | "all"
    border_penalty: float = 1.0


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Alanı `min_area`'dan küçük bağlantılı bölgeleri siler.

    skimage.morphology.remove_small_objects yerine doğrudan OpenCV ile
    yazıldı: skimage 0.26'da `min_size` parametresi deprecate edildi ve
    eşik semantiği değişti, sürümler arası davranış farkı istemiyoruz.
    """
    if min_area <= 1 or not mask.any():
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8)


def fill_interior(mask: np.ndarray) -> np.ndarray:
    """Maskedeki tüm kapalı delikleri doldurur.

    Bir lezyon tek parça, dolu bir bölgedir — maskenin içinde delik olması
    her zaman segmentasyon hatasıdır. Özellikle adaptif eşiklemede lezyonun
    yalnızca kenarı yakalanır (iç kısımda yerel kontrast yoktur), bu adım
    o halkayı dolu bölgeye çevirir.
    """
    if not mask.any():
        return mask
    return binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def _border_coverage(region: np.ndarray) -> float:
    """Bölgenin görüntü kenar çerçevesinin ne kadarını kapladığını döndürür.

    Vinyet/mercek halkası gibi artefaktlar çerçevenin tamamını dolaşır
    (≈1.0); gerçek bir lezyon kenara değse bile küçük bir kısmını kaplar.
    """
    top, bottom = region[0, :], region[-1, :]
    left, right = region[:, 0], region[:, -1]
    border = np.concatenate([top, bottom, left, right])
    return float(border.mean()) if border.size else 0.0


def select_region(
    mask: np.ndarray, strategy: str = "center", border_penalty: float = 1.0
) -> np.ndarray:
    """Birden çok bölge varsa lezyon olanı seçer.

    Dermoskopi görüntülerinde ilgilenilen lezyon kadraja ortalanmış olur, bu
    yüzden skor hem alanı hem merkeze yakınlığı hesaba katar.

    Merkez mesafesi tek başına yeterli değil: vinyetten doğan halka şeklindeki
    bir bölgenin ağırlık merkezi de tam görüntü merkezinde çıkar ve alanı
    büyük olduğu için lezyonu yenir. Bu yüzden çerçeveyi dolaşan bölgeler
    `border_penalty` ile cezalandırılıyor.
    """
    if strategy == "all" or not mask.any():
        return mask

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 2:  # 0 = arka plan; tek bölge varsa seçim gereksiz
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)

    if strategy == "largest":
        scores = areas.copy()
    elif strategy == "center":
        height, width = mask.shape
        center = np.array([width / 2.0, height / 2.0])
        half_diagonal = 0.5 * float(np.hypot(height, width))
        distances = np.linalg.norm(centroids[1:] - center, axis=1)
        # Merkezden uzaklaştıkça ceza artar; yarım köşegende skor ~1/2'ye iner.
        scores = areas / (1.0 + (distances / half_diagonal) ** 2)
    else:
        raise ValueError(f"Bilinmeyen secim stratejisi: {strategy}")

    if border_penalty > 0:
        for index in range(1, count):
            coverage = _border_coverage(labels == index)
            scores[index - 1] *= max(1.0 - border_penalty * coverage, 0.0)

    if not np.any(scores > 0):  # Tüm adaylar cezalandırıldıysa alana geri dön
        scores = areas

    best = 1 + int(np.argmax(scores))
    return (labels == best).astype(np.uint8)


def postprocess(mask: np.ndarray, config: PostprocessConfig | None = None) -> np.ndarray:
    """Maskeyi temizler ve tek bölgeye indirir."""
    config = config or PostprocessConfig()
    mask = (mask > 0).astype(np.uint8)
    if not mask.any():
        return mask

    pixels = mask.size

    # Önce açma: ince köprüleri kopar, böylece bitişik gürültü lezyona
    # bağlanıp tek bir büyük bölge gibi görünmesin.
    if config.opening_size >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.opening_size, config.opening_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = remove_small_regions(mask, int(pixels * config.min_area_ratio))

    # Doldurma bölge seçiminden ÖNCE yapılır: adaptif eşiklemenin ürettiği
    # ince halka, doldurulmazsa alanı küçük kalır ve seçimde kaybeder.
    if config.fill_holes:
        mask = fill_interior(mask)

    if config.closing_size >= 3:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.closing_size, config.closing_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if config.fill_holes:
            # Kapama yeni kapalı boşluklar bırakabilir.
            mask = fill_interior(mask)

    return select_region(mask, config.select, config.border_penalty)
