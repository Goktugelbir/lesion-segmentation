"""Veri seti indirmeden pipeline'ı deneme/doğrulama betiği.

Sentetik dermoskopi görüntüleri üretir, isteğe bağlı olarak diske yazar ve
dört yöntemi karşılaştıran bir figür kaydeder.

Kullanım:
    python scripts/demo_synthetic.py
    python scripts/demo_synthetic.py --write data_synthetic --count 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import build_pipelines  # noqa: E402
from src.synthetic import synthetic_batch  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=6, help="Üretilecek örnek sayısı.")
    parser.add_argument("--seed", type=int, default=0, help="Rastgelelik tohumu.")
    parser.add_argument(
        "--write",
        default=None,
        help="Verilirse örnekler bu klasöre images/ ve masks/ olarak yazılır.",
    )
    parser.add_argument(
        "--figure",
        default="synthetic_comparison.png",
        help="Karşılaştırma figürünün kaydedileceği dosya ('' ile kapatılır).",
    )
    return parser.parse_args(argv)


def write_dataset(samples, root: Path) -> None:
    """Sentetik örnekleri LesionDataset'in okuyabileceği yapıda kaydeder."""
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index, (image, mask) in enumerate(samples):
        name = f"synthetic_{index:04d}"
        cv2.imwrite(str(image_dir / f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask * 255)

    print(f"{len(samples)} ornek yazildi: {root.resolve()}")


def main(argv=None) -> int:
    args = parse_args(argv)
    samples = list(synthetic_batch(count=args.count, seed=args.seed))

    if args.write:
        write_dataset(samples, Path(args.write))

    pipelines = build_pipelines()
    scores: dict[str, list[float]] = {name: [] for name in pipelines}

    for image, truth in samples:
        for name, pipeline in pipelines.items():
            scores[name].append(pipeline.evaluate(image, truth)["dice"])

    print(f"\n{len(samples)} sentetik ornek, Dice:")
    print(f"{'yontem':<12}{'ortalama':>10}{'std':>10}{'en kotu':>10}")
    print("-" * 42)
    for name, values in sorted(scores.items(), key=lambda kv: -float(np.mean(kv[1]))):
        array = np.asarray(values, dtype=float)
        print(f"{name:<12}{array.mean():>10.4f}{array.std():>10.4f}{array.min():>10.4f}")

    print(
        "\nUyari: sentetik goruntuler gercek dermoskopiden belirgin olarak kolaydir."
        "\nBu sayilar kodun calistigini gosterir, gercek veri performansini gostermez."
    )

    if args.figure:
        # Ekran olmayan ortamda da kaydedebilmek için Agg arka ucu.
        import matplotlib

        matplotlib.use("Agg")
        from src.viz import compare_methods

        image, truth = samples[0]
        figure = compare_methods(image, truth, pipelines)
        figure.savefig(args.figure, dpi=110, bbox_inches="tight")
        print(f"Figur kaydedildi: {Path(args.figure).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
