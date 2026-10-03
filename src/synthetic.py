"""Sentetik dermoskopi görüntüsü üretimi.

Veri seti indirilmeden pipeline'ın çalıştığını doğrulamak için gerekli:
testler ve notebook'un ilk hücresi gerçek veri olmadan da çalışabiliyor.
Üretilen görüntüler gerçek lezyonların yerine geçmez, yalnızca kodun
beklenen davranışını sınar.
"""

from __future__ import annotations

import cv2
import numpy as np


def synthetic_lesion(
    size: tuple[int, int] = (320, 320),
    lesion_radius: float = 0.22,
    offset: tuple[float, float] = (0.0, 0.0),
    hairs: int = 12,
    vignette: bool = True,
    noise: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Koyu bir lezyon, kıllar ve vinyet içeren görüntü + gerçek maske üretir.

    Args:
        size: (yükseklik, genişlik).
        lesion_radius: kısa kenara oranla lezyon yarıçapı.
        offset: lezyon merkezinin görüntü merkezinden kaydırılması (oran).
        hairs: çizilecek kıl sayısı.
        vignette: köşelerde karartma uygulanıp uygulanmayacağı.
        noise: eklenecek Gauss gürültüsünün standart sapması.
        seed: rastgelelik tohumu.

    Returns:
        (RGB uint8 görüntü, 0/1 maske)
    """
    rng = np.random.default_rng(seed)
    height, width = size

    # Deri zemini: hafif pembe-bej, yumuşak doku varyasyonu ile.
    skin = np.zeros((height, width, 3), dtype=np.float32)
    skin[:, :] = (214.0, 172.0, 150.0)
    texture = rng.normal(0.0, 6.0, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 6.0)
    skin += texture[:, :, None]

    # Lezyon: düzensiz kenarlı, deriden belirgin koyu bir bölge.
    center = (
        int(width * (0.5 + offset[1])),
        int(height * (0.5 + offset[0])),
    )
    radius = lesion_radius * min(height, width)

    ys, xs = np.mgrid[0:height, 0:width]
    angle = np.arctan2(ys - center[1], xs - center[0])
    # Açıya bağlı dalgalanma kenarı daireden uzaklaştırır.
    wobble = 1.0 + 0.18 * np.sin(3 * angle + 0.7) + 0.09 * np.cos(5 * angle)
    distance = np.hypot(xs - center[0], ys - center[1])
    mask = (distance < radius * wobble).astype(np.uint8)

    # Kenarın yumuşak geçmesi için maskeyi bulanıklaştırıp alfa olarak kullan.
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)[:, :, None]
    lesion_color = np.array([92.0, 62.0, 58.0], dtype=np.float32)
    image = skin * (1.0 - alpha) + lesion_color * alpha

    # Kıllar: ince, koyu, görüntüyü boydan boya geçen eğriler.
    if hairs > 0:
        layer = image.copy()
        for _ in range(hairs):
            start = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            end = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            mid = (
                (start[0] + end[0]) // 2 + int(rng.integers(-40, 41)),
                (start[1] + end[1]) // 2 + int(rng.integers(-40, 41)),
            )
            curve = np.array([start, mid, end], dtype=np.int32)
            cv2.polylines(
                layer,
                [curve],
                isClosed=False,
                color=(38.0, 28.0, 24.0),
                thickness=int(rng.integers(1, 3)),
                lineType=cv2.LINE_AA,
            )
        image = layer

    if vignette:
        # Merkezden uzaklaştıkça koyulaşan çarpan.
        normalized = np.hypot(
            (xs - width / 2) / (width / 2), (ys - height / 2) / (height / 2)
        )
        falloff = np.clip(1.0 - 0.45 * normalized**2, 0.0, 1.0).astype(np.float32)
        image *= falloff[:, :, None]

    if noise > 0:
        image += rng.normal(0.0, noise, image.shape).astype(np.float32)

    return np.clip(image, 0, 255).astype(np.uint8), mask


def synthetic_batch(count: int = 8, seed: int = 0, **kwargs):
    """Farklı boyut/konumda birden çok sentetik örnek üretir."""
    rng = np.random.default_rng(seed)
    for index in range(count):
        yield synthetic_lesion(
            lesion_radius=float(rng.uniform(0.14, 0.30)),
            offset=(float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.08, 0.08))),
            hairs=int(rng.integers(0, 20)),
            seed=seed + index,
            **kwargs,
        )
