"""Ön işleme + segmentasyon + son işlemeyi tek nesnede birleştirir.

Değerlendirme betiği, notebook ve testler aynı yolu kullansın diye tek giriş
noktası: `Pipeline.run(image)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import align, compute_metrics
from .postprocess import PostprocessConfig, postprocess
from .preprocess import PreprocessConfig, preprocess
from .segment import SEGMENTERS


@dataclass
class Pipeline:
    """Tek bir segmentasyon yapılandırması.

    Attributes:
        method: SEGMENTERS içindeki yöntem adı.
        pre: ön işleme yapılandırması.
        post: son işleme yapılandırması.
        segmenter_kwargs: yönteme özel ek parametreler (ör. kmeans clusters).
    """

    method: str = "otsu"
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    post: PostprocessConfig = field(default_factory=PostprocessConfig)
    segmenter_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in SEGMENTERS:
            available = ", ".join(sorted(SEGMENTERS))
            raise ValueError(f"Bilinmeyen yontem '{self.method}'. Secenekler: {available}")

    def run(self, image: np.ndarray) -> dict:
        """Görüntüyü işler; maskeyi ve ara adımları döndürür."""
        stages = preprocess(image, self.pre)

        segmenter = SEGMENTERS[self.method]
        raw = segmenter(stages["result"], stages["fov"], **self.segmenter_kwargs)

        stages["raw_mask"] = raw
        stages["mask"] = postprocess(raw, self.post)
        return stages

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Sadece son maskeyi döndürür."""
        return self.run(image)["mask"]

    def evaluate(
        self, image: np.ndarray, truth: np.ndarray, with_boundary: bool = True
    ) -> dict[str, float]:
        """Tek görüntü için metrikleri hesaplar."""
        prediction = align(self.predict(image), truth)
        return compute_metrics(prediction, truth, with_boundary=with_boundary)


def default_pipeline(method: str) -> Pipeline:
    """Yöntemin varsayılan pipeline'ı.

    Dört temel yöntem aynı ön/son işlemeyi paylaşır (adil karşılaştırma).
    `otsu_plus` ise kendine göre ayarlanmış zinciri kullanır: ablasyonda CLAHE
    global eşiklemeye zarar verdiği için kapalı, vinyet FOV'dan çıkarılıyor ve
    son maske eksik bölütlemeyi telafi etmek için genişletiliyor. Parametreler
    raporlanan 1000 görüntüden ayrı bir alt kümede (5000–5399) seçildi.
    """
    if method == "otsu_plus":
        return Pipeline(
            method=method,
            pre=PreprocessConfig(clahe=False, remove_vignette=True),
            post=PostprocessConfig(final_dilation=13),
        )
    return Pipeline(method=method)


def build_pipelines(methods=None) -> dict[str, Pipeline]:
    """Karşılaştırma için varsayılan pipeline kümesi üretir."""
    methods = list(SEGMENTERS) if methods is None else list(methods)
    return {name: default_pipeline(name) for name in methods}


def ablation_pipelines(method: str = "otsu") -> dict[str, Pipeline]:
    """Ön işleme adımlarının katkısını ölçmek için pipeline varyantları.

    Her varyantta tek bir adım kapatılır; metrik düşüşü o adımın katkısıdır.
    """
    variants = {
        "tam": PreprocessConfig(),
        "sac_temizleme_yok": PreprocessConfig(hair_removal=False),
        "golge_duzeltme_yok": PreprocessConfig(shading_correction=False),
        "clahe_yok": PreprocessConfig(clahe=False),
        "fov_yok": PreprocessConfig(restrict_to_fov=False),
        "on_isleme_yok": PreprocessConfig(
            hair_removal=False,
            shading_correction=False,
            clahe=False,
            median_blur=0,
            restrict_to_fov=False,
        ),
    }
    return {name: Pipeline(method=method, pre=config) for name, config in variants.items()}
