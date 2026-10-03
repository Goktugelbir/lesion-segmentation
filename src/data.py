"""Veri seti keşfi ve görüntü/maske yükleme.

Hem Kaggle ortamındaki `/kaggle/input/...` yollarını hem de yerel bir klasörü
otomatik bulur; böylece notebook'ta yol değiştirmek gerekmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
MASK_SUFFIXES = (".png", ".jpg", ".bmp")

# Veri setinin bulunabileceği aday kökler. İlk eşleşen kullanılır.
CANDIDATE_ROOTS = (
    "data",
    "../data",
    "/kaggle/input/skin-cancer-lesions-segmentation/data",
    "/kaggle/input/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation/data",
)

# `images` ve `masks` klasörlerinin alabileceği isimler.
IMAGE_DIR_NAMES = ("images", "image", "imgs", "inputs")
MASK_DIR_NAMES = ("masks", "mask", "labels", "ground_truth", "gt")


class DatasetNotFound(RuntimeError):
    """Veri seti kökü bulunamadığında atılır."""


def _first_existing_subdir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def find_dataset(root: str | Path | None = None) -> tuple[Path, Path]:
    """Görüntü ve maske klasörlerini bulur.

    `root` verilirse sadece orada arar, verilmezse CANDIDATE_ROOTS listesini
    sırayla dener. Kökün kendisi doğrudan `images`/`masks` içeriyor olabilir ya
    da bir alt klasörde (ör. `root/data/images`) saklanıyor olabilir.
    """
    roots = [Path(root)] if root is not None else [Path(p) for p in CANDIDATE_ROOTS]

    for candidate in roots:
        if not candidate.is_dir():
            continue
        # Doğrudan root/images + root/masks
        image_dir = _first_existing_subdir(candidate, IMAGE_DIR_NAMES)
        mask_dir = _first_existing_subdir(candidate, MASK_DIR_NAMES)
        if image_dir and mask_dir:
            return image_dir, mask_dir
        # Bir seviye daha derinde ara (ör. root/data/images)
        for child in sorted(p for p in candidate.iterdir() if p.is_dir()):
            image_dir = _first_existing_subdir(child, IMAGE_DIR_NAMES)
            mask_dir = _first_existing_subdir(child, MASK_DIR_NAMES)
            if image_dir and mask_dir:
                return image_dir, mask_dir

    searched = ", ".join(str(p) for p in roots)
    raise DatasetNotFound(
        f"images/masks klasorleri bulunamadi. Arananlar: {searched}. "
        "Veri setini indirip --data ile yolunu verin."
    )


def load_image(path: str | Path) -> np.ndarray:
    """Görüntüyü RGB uint8 olarak okur."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    """Maskeyi 0/1 değerli uint8 olarak okur.

    Maskeler bazı veri setlerinde 0/255, bazılarında 0/1 kaydedilir; eşik
    maksimum değere göre seçilir, böylece ikisi de doğru okunur.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadi: {path}")
    threshold = 127 if mask.max() > 1 else 0
    return (mask > threshold).astype(np.uint8)


@dataclass(frozen=True)
class Sample:
    """Tek bir görüntü-maske çifti."""

    name: str
    image_path: Path
    mask_path: Path

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        return load_image(self.image_path), load_mask(self.mask_path)


class LesionDataset:
    """Eşleşen görüntü-maske çiftlerinin indekslenebilir listesi.

    Eşleştirme dosya adının gövdesine göre yapılır: `ISIC_0024306.jpg` ile
    `ISIC_0024306.png` aynı örnektir. Maskesi olmayan görüntüler sessizce
    atlanır, sayısı `skipped` içinde tutulur.
    """

    def __init__(self, root: str | Path | None = None, limit: int | None = None):
        self.image_dir, self.mask_dir = find_dataset(root)

        masks_by_stem = {
            path.stem: path
            for suffix in MASK_SUFFIXES
            for path in self.mask_dir.glob(f"*{suffix}")
        }

        image_paths = sorted(
            path
            for suffix in IMAGE_SUFFIXES
            for path in self.image_dir.glob(f"*{suffix}")
        )

        samples: list[Sample] = []
        skipped = 0
        for image_path in image_paths:
            mask_path = masks_by_stem.get(image_path.stem)
            if mask_path is None:
                skipped += 1
                continue
            samples.append(Sample(image_path.stem, image_path, mask_path))

        self.skipped = skipped
        self.samples = samples[:limit] if limit is not None else samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __iter__(self):
        return iter(self.samples)

    def subset(self, indices) -> list[Sample]:
        """Belirli indekslerdeki örnekleri döndürür (görselleştirme için)."""
        return [self.samples[i] for i in indices]

    def __repr__(self) -> str:
        return (
            f"LesionDataset({len(self)} ornek, atlanan={self.skipped}, "
            f"images={self.image_dir}, masks={self.mask_dir})"
        )
