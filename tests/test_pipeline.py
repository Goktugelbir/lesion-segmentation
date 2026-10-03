"""Pipeline ve metriklerin doğruluk testleri.

Gerçek veri seti olmadan çalışır: sentetik görüntüler kullanılır.
Çalıştırmak için:  python -m pytest tests -q   (ya da python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import align, compute_metrics  # noqa: E402
from src.pipeline import Pipeline, build_pipelines  # noqa: E402
from src.postprocess import PostprocessConfig, fill_interior, select_region  # noqa: E402
from src.preprocess import hair_mask, line_kernel, preprocess  # noqa: E402
from src.segment import SEGMENTERS  # noqa: E402
from src.synthetic import synthetic_batch, synthetic_lesion  # noqa: E402


# --- metrikler ---------------------------------------------------------------

def test_metrics_perfect_match():
    """Aynı maskeler için IoU ve Dice 1, mesafe metrikleri 0 olmalı."""
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 1

    result = compute_metrics(mask, mask)
    assert result["iou"] == 1.0
    assert result["dice"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["xor_error"] == 0.0
    assert result["hd95"] == 0.0
    assert result["assd"] == 0.0


def test_metrics_disjoint_match():
    """Hiç örtüşmeyen maskelerde örtüşme metrikleri 0 olmalı."""
    truth = np.zeros((40, 40), np.uint8)
    truth[0:10, 0:10] = 1
    prediction = np.zeros((40, 40), np.uint8)
    prediction[30:40, 30:40] = 1

    result = compute_metrics(prediction, truth)
    assert result["iou"] == 0.0
    assert result["dice"] == 0.0
    assert result["hd95"] > 0.0


def test_metrics_known_values():
    """Elle hesaplanabilir bir örnekte beklenen değerler çıkmalı."""
    truth = np.zeros((10, 10), np.uint8)
    truth[:, :5] = 1  # 50 piksel
    prediction = np.zeros((10, 10), np.uint8)
    prediction[:, :3] = 1  # 30 piksel, tamamı doğru

    result = compute_metrics(prediction, truth, with_boundary=False)
    # TP=30, FP=0, FN=20 -> IoU=30/50, Dice=60/80
    assert abs(result["iou"] - 0.6) < 1e-9
    assert abs(result["dice"] - 0.75) < 1e-9
    assert abs(result["sensitivity"] - 0.6) < 1e-9
    assert result["precision"] == 1.0
    assert abs(result["xor_error"] - 0.4) < 1e-9


def test_metrics_empty_prediction_is_nan_not_zero():
    """Boş tahminde tanımsız metrikler NaN dönmeli, 0 değil.

    0 döndürmek "ölçüldü ve sıfır" ile "hesaplanamaz" durumlarını
    karıştırırdı; ortalamalar bundan yanlış etkilenir.
    """
    truth = np.zeros((20, 20), np.uint8)
    truth[5:15, 5:15] = 1
    empty = np.zeros((20, 20), np.uint8)

    result = compute_metrics(empty, truth)
    assert result["dice"] == 0.0  # 2*0/(0+0+100) tanımlı
    assert np.isnan(result["precision"])  # TP+FP = 0 -> tanımsız
    assert np.isnan(result["hd95"])  # sınır yok


def test_align_resizes_to_reference():
    reference = np.zeros((60, 80), np.uint8)
    mask = np.ones((30, 40), np.uint8)
    aligned = align(mask, reference)
    assert aligned.shape == reference.shape
    assert set(np.unique(aligned)) <= {0, 1}


# --- ön işleme ---------------------------------------------------------------

def test_line_kernel_angles_differ():
    """0 ve 90 derece çekirdekleri birbirinin devriği olmalı."""
    horizontal = line_kernel(9, 0)
    vertical = line_kernel(9, 90)
    assert horizontal.shape == (1, 9)
    assert vertical.shape == (9, 1)
    for angle in (45, 135):
        kernel = line_kernel(9, angle)
        assert kernel.shape == (9, 9)
        assert kernel.any(), f"{angle} derece cekirdek bos"


def test_hair_mask_finds_drawn_hairs():
    """Kıllı görüntüde maske dolu, kılsız görüntüde neredeyse boş olmalı."""
    with_hairs, _ = synthetic_lesion(hairs=25, seed=5)
    without_hairs, _ = synthetic_lesion(hairs=0, seed=5)

    detected = hair_mask(with_hairs).mean()
    false_positive = hair_mask(without_hairs).mean()

    assert detected > 0.01, f"kil tespit edilemedi: {detected:.4f}"
    assert detected > false_positive * 3, (
        f"kilsiz goruntude de benzer oranda tespit var: {detected:.4f} vs {false_positive:.4f}"
    )


def test_preprocess_returns_expected_stages():
    image, _ = synthetic_lesion(seed=2)
    stages = preprocess(image)

    for key in ("original", "hair_mask", "shade_corrected", "enhanced", "fov", "result"):
        assert key in stages, f"eksik asama: {key}"
    assert stages["result"].shape == image.shape
    assert stages["fov"].shape == image.shape[:2]


def test_shading_correction_preserves_lesion_contrast():
    """Gölge düzeltmesi lezyon ile deri arasındaki kontrastı yok etmemeli.

    Aydınlatmayı geniş bulanıklaştırma ile tahmin eden bir uygulama büyük
    lezyonları da "aydınlatma" sanıp düzleştiriyordu; bu test o gerilemeyi
    yakalar.
    """
    import cv2

    from src.preprocess import correct_shading

    image, truth = synthetic_lesion(lesion_radius=0.3, hairs=0, seed=7)
    corrected = correct_shading(image)

    gray = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY).astype(float)
    lesion_mean = gray[truth > 0].mean()
    skin_mean = gray[truth == 0].mean()

    assert skin_mean - lesion_mean > 30, (
        f"lezyon-deri kontrasti kayboldu: deri={skin_mean:.1f} lezyon={lesion_mean:.1f}"
    )


# --- son işleme --------------------------------------------------------------

def test_fill_interior_fills_ring():
    """İçi boş halka dolu diske dönüşmeli."""
    import cv2

    radius = 30
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), radius, 1, thickness=4)

    filled = fill_interior(mask)
    assert filled[50, 50] == 1, "halkanin ici dolmadi"
    # Sonuç yarıçapı ~radius olan dolu bir disk olmalı (halka kalınlığı payı
    # bırakarak alt sınır veriyoruz).
    expected_area = np.pi * (radius - 2) ** 2
    assert filled.sum() > expected_area, f"alan beklenenden kucuk: {filled.sum()}"


def test_select_region_rejects_border_ring():
    """Çerçeveyi dolaşan artefakt yerine merkezdeki lezyon seçilmeli.

    Halkanın ağırlık merkezi de görüntü merkezindedir ve alanı daha büyüktür;
    yalnızca merkeze yakınlığa bakan bir seçim halkayı seçerdi.
    """
    mask = np.zeros((120, 120), np.uint8)
    # Kenarları dolaşan kalın halka (vinyet artefaktı benzeri)
    mask[:12, :] = 1
    mask[-12:, :] = 1
    mask[:, :12] = 1
    mask[:, -12:] = 1
    # Merkezdeki asıl lezyon (daha küçük alan)
    mask[50:70, 50:70] = 1

    selected = select_region(mask, strategy="center", border_penalty=1.0)
    assert selected[60, 60] == 1, "merkezdeki lezyon secilmedi"
    assert selected[2, 2] == 0, "kenar halkasi secildi"


def test_select_region_largest_strategy():
    mask = np.zeros((80, 80), np.uint8)
    mask[5:10, 5:10] = 1  # küçük
    mask[30:60, 30:60] = 1  # büyük
    selected = select_region(mask, strategy="largest", border_penalty=0.0)
    assert selected[45, 45] == 1
    assert selected[7, 7] == 0


def test_postprocess_removes_speckles():
    mask = np.zeros((200, 200), np.uint8)
    mask[80:120, 80:120] = 1  # gerçek bölge
    rng = np.random.default_rng(0)
    for _ in range(40):  # tekil gürültü pikselleri
        y, x = rng.integers(0, 200, 2)
        mask[y, x] = 1

    cleaned = postprocess_default(mask)
    assert cleaned[100, 100] == 1
    # Gürültü pikselleri temizlenmiş olmalı: sonuç tek bitişik bölge
    import cv2

    count, _ = cv2.connectedComponents(cleaned)
    assert count == 2, f"beklenen 1 bolge, bulunan {count - 1}"


def postprocess_default(mask):
    from src.postprocess import postprocess

    return postprocess(mask, PostprocessConfig())


# --- uçtan uca ---------------------------------------------------------------

def test_all_methods_segment_synthetic_lesion():
    """Dört yöntem de sentetik lezyonu makul doğrulukla bulmalı.

    Sentetik görüntü gerçek dermoskopiden kolaydır; buradaki eşik bir başarı
    iddiası değil, pipeline'ın bozulmadığını gösteren alt sınır.
    """
    scores = {name: [] for name in SEGMENTERS}

    for image, truth in synthetic_batch(count=4, seed=11):
        for name, pipeline in build_pipelines().items():
            scores[name].append(pipeline.evaluate(image, truth)["dice"])

    for name, values in scores.items():
        mean_dice = float(np.mean(values))
        assert mean_dice > 0.8, f"{name} sentetik veride basarisiz: Dice={mean_dice:.3f}"


def test_pipeline_rejects_unknown_method():
    try:
        Pipeline(method="yok-boyle-bir-yontem")
    except ValueError as error:
        assert "Bilinmeyen" in str(error)
    else:
        raise AssertionError("gecersiz yontem icin hata beklendi")


def test_pipeline_handles_blank_image():
    """Tamamen düz bir görüntüde çökmemeli, boş maske dönmeli."""
    blank = np.full((64, 64, 3), 200, np.uint8)
    for name, pipeline in build_pipelines().items():
        mask = pipeline.predict(blank)
        assert mask.shape == (64, 64), name
        assert set(np.unique(mask)) <= {0, 1}, name


def test_mask_is_binary_and_single_region():
    image, _ = synthetic_lesion(seed=4)
    mask = Pipeline(method="otsu").predict(image)
    assert set(np.unique(mask)) <= {0, 1}

    import cv2

    count, _ = cv2.connectedComponents(mask)
    assert count <= 2, "varsayilan ayarda tek bolge beklenir"


if __name__ == "__main__":
    # pytest kurulu olmasa da çalışsın.
    failed = 0
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} test gecti")
    raise SystemExit(1 if failed else 0)
