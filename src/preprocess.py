"""Ön işleme: saç temizleme, gölge düzeltme, kontrast iyileştirme.

Dermoskopi görüntülerinde segmentasyonu bozan üç ana artefakt var:
kıl/tüy, köşelerde vinyet (gölgelenme) ve düşük yerel kontrast. Buradaki
fonksiyonlar bunların her birini ayrı ayrı ele alır.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def line_kernel(length: int, angle: int) -> np.ndarray:
    """Verilen açıda doğrusal yapısal eleman üretir.

    Kıllar ince ve uzun yapılar olduğu için, farklı açılardaki doğrusal
    elemanlarla yapılan morfolojik kapama onları belirginleştirir.
    """
    if angle % 180 == 0:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    if angle % 180 == 90:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))

    # Ara açılar: yatay çizgiyi döndürerek elde edilir.
    kernel = np.zeros((length, length), dtype=np.uint8)
    kernel[length // 2, :] = 1
    center = (length / 2 - 0.5, length / 2 - 0.5)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(kernel, rotation, (length, length), flags=cv2.INTER_NEAREST)
    if not rotated.any():  # Çok küçük çekirdeklerde dönüş her şeyi silebilir
        rotated[length // 2, :] = 1
    return rotated


def hair_mask(
    image: np.ndarray,
    kernel_length: int = 17,
    threshold: int = 10,
    min_hair_area: int = 25,
) -> np.ndarray:
    """Kıl piksellerini işaretleyen ikili maske üretir (DullRazor yaklaşımı).

    Kıllar çevrelerindeki deriden koyu olduğu için, morfolojik kapama onları
    "doldurur". Kapanmış görüntü ile orijinal arasındaki fark kılların
    bulunduğu yerde büyük olur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Dört yönde kapama; kıl hangi açıda olursa olsun en az birinde yakalanır.
    closings = [
        cv2.morphologyEx(gray, cv2.MORPH_CLOSE, line_kernel(kernel_length, angle))
        for angle in (0, 45, 90, 135)
    ]
    strongest = np.maximum.reduce(closings)

    response = cv2.absdiff(strongest, gray)
    mask = (response > threshold).astype(np.uint8)

    # Eşikleme kıl olmayan tekil pikselleri de yakalar; alanı küçük olanları at.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_hair_area:
            cleaned[labels == label] = 1

    # İnpaint'in kıl kenarlarını da kapatması için biraz genişlet.
    return cv2.dilate(cleaned, np.ones((3, 3), np.uint8), iterations=1)


def remove_hair(
    image: np.ndarray,
    kernel_length: int = 17,
    threshold: int = 10,
    inpaint_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Kılları tespit edip inpaint ile doldurur.

    Returns:
        (temizlenmiş görüntü, kullanılan kıl maskesi)
    """
    mask = hair_mask(image, kernel_length, threshold)
    if not mask.any():
        return image.copy(), mask

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), mask


def correct_shading(
    image: np.ndarray,
    fov: np.ndarray | None = None,
    skin_percentile: float = 50.0,
    subsample: int = 4,
) -> np.ndarray:
    """Köşelerdeki vinyet/gölgelenmeyi bastırır.

    Aydınlatma alanı, parlaklık kanalına ikinci derece bir yüzey (quadratic)
    oturtularak tahmin edilir. Vinyet merkezden uzaklıkla yaklaşık kuadratik
    azaldığı için bu model yeterli.

    Yüzey yalnızca *deri* piksellerinden (parlaklığı medyanın üzerinde olanlar)
    hesaplanır. Bu önemli: büyük bir lezyon da koyu olduğu için fit'e dahil
    edilirse aydınlatma tahmini lezyonun üzerinde düşer, bölme işlemi lezyonu
    aydınlatır ve segmentasyon için gereken kontrast kaybolur. Bulanıklaştırma
    tabanlı (ör. çok geniş Gaussian) tahminlerde tam bu sorun oluşuyor.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    height, width = luminance.shape

    # Koordinatları [-1, 1] aralığına normalize etmek en küçük kareler
    # çözümünün sayısal koşullanmasını iyileştirir.
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    x_norm = xs / max(width - 1, 1) * 2.0 - 1.0
    y_norm = ys / max(height - 1, 1) * 2.0 - 1.0
    basis = np.stack(
        [
            np.ones_like(x_norm),
            x_norm,
            y_norm,
            x_norm * x_norm,
            x_norm * y_norm,
            y_norm * y_norm,
        ],
        axis=-1,
    )

    valid = np.ones(luminance.shape, dtype=bool) if fov is None else fov > 0
    if valid.sum() < 50:
        return image.copy()

    threshold = float(np.percentile(luminance[valid], skin_percentile))
    skin = valid & (luminance >= threshold)
    if skin.sum() < basis.shape[-1] * 10:
        return image.copy()

    # Fit için her `subsample` pikselden birini kullanmak sonucu değiştirmeden
    # işlemi hızlandırır (yüzey çok düşük dereceli).
    sampled = np.zeros_like(skin)
    sampled[::subsample, ::subsample] = True
    fit_mask = skin & sampled
    if fit_mask.sum() < basis.shape[-1] * 10:
        fit_mask = skin

    coefficients, *_ = np.linalg.lstsq(
        basis[fit_mask], luminance[fit_mask], rcond=None
    )
    illumination = np.maximum(basis @ coefficients, 1.0)

    # Deri parlaklığının ortalamasına yeniden ölçekle: genel ton korunur,
    # yalnızca mekânsal eğim giderilir.
    corrected = luminance / illumination * float(luminance[skin].mean())

    lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8
) -> np.ndarray:
    """LAB uzayında CLAHE ile yerel kontrastı artırır.

    CLAHE yalnızca L (parlaklık) kanalına uygulanır; a/b kanallarına
    dokunulmadığı için lezyonun renk tonu bozulmaz.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def field_of_view(image: np.ndarray, dark_threshold: int = 25) -> np.ndarray:
    """Dermatoskop merceğinin dışındaki siyah çerçeveyi maskeler.

    Bazı görüntülerde kenarlarda tamamen siyah bir halka olur. Lezyon da koyu
    olduğu için eşikleme bu halkayı lezyon sanar; geçerli bölgeyi döndürüp
    segmentasyonu oraya sınırlandırıyoruz.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    valid = (gray > dark_threshold).astype(np.uint8)

    # Küçük delikleri kapat, sonra en büyük bağlantılı bölgeyi al.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    if count <= 1:
        return np.ones(gray.shape, dtype=np.uint8)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest).astype(np.uint8)

    # Siyah halka yoksa bütün görüntü geçerlidir; gereksiz kırpma yapmayalım.
    if region.mean() > 0.98:
        return np.ones(gray.shape, dtype=np.uint8)

    # Kenardan biraz içeri çekerek halkanın yumuşak geçişini de dışarıda bırak.
    return cv2.erode(region, kernel, iterations=1)


@dataclass
class PreprocessConfig:
    """Ön işleme adımlarının açık/kapalı durumu ve parametreleri.

    Adımları tek tek kapatabilmek ablasyon çalışması için gerekli: hangi
    adımın metriğe ne kadar katkı verdiğini ölçebilmek istiyoruz.
    """

    hair_removal: bool = True
    hair_kernel_length: int = 17
    hair_threshold: int = 10
    shading_correction: bool = True
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    median_blur: int = 5  # 0 => kapalı
    restrict_to_fov: bool = True


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> dict:
    """Ön işleme zincirini uygular ve ara sonuçları da döndürür.

    Ara adımları saklamak hem görselleştirme hem de hata ayıklama için
    gerekli; zincirin neresinde bozulduğunu görmek istiyoruz.
    """
    config = config or PreprocessConfig()
    stages: dict[str, np.ndarray] = {"original": image}

    # Geçerli bölge en başta hesaplanır; gölge düzeltmesi de buna ihtiyaç duyar
    # (mercek dışındaki siyah piksellerin aydınlatma fit'ini bozmaması için).
    stages["fov"] = (
        field_of_view(image)
        if config.restrict_to_fov
        else np.ones(image.shape[:2], dtype=np.uint8)
    )

    current = image
    if config.hair_removal:
        current, hairs = remove_hair(
            current,
            kernel_length=config.hair_kernel_length,
            threshold=config.hair_threshold,
        )
        stages["hair_mask"] = hairs
        stages["dehaired"] = current

    if config.median_blur >= 3:
        # Tek sayı olmak zorunda; çift verilirse bir artırıyoruz.
        size = config.median_blur + (config.median_blur % 2 == 0)
        current = cv2.medianBlur(current, size)
        stages["denoised"] = current

    if config.shading_correction:
        current = correct_shading(current, fov=stages["fov"])
        stages["shade_corrected"] = current

    if config.clahe:
        current = enhance_contrast(current, clip_limit=config.clahe_clip_limit)
        stages["enhanced"] = current

    stages["result"] = current
    return stages
