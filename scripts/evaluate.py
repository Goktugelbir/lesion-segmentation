"""Veri seti üzerinde yöntemleri karşılaştıran değerlendirme betiği.

Kullanım:
    python scripts/evaluate.py --data data --limit 300
    python scripts/evaluate.py --methods otsu,kmeans --no-boundary --out sonuc.csv
    python scripts/evaluate.py --ablation --limit 200

Çıktı: görüntü başına metrikleri içeren CSV + konsola özet tablo ve
yöntemler arası eşleştirilmiş istatistiksel karşılaştırma.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DatasetNotFound, LesionDataset  # noqa: E402
from src.metrics import LOWER_IS_BETTER, METRIC_NAMES, align, compute_metrics  # noqa: E402
from src.pipeline import ablation_pipelines, build_pipelines  # noqa: E402
from src.segment import SEGMENTERS  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        default=None,
        help="Veri seti kökü. Verilmezse bilinen yollarda otomatik aranır.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(SEGMENTERS),
        help=f"Virgülle ayrılmış yöntem listesi. Seçenekler: {', '.join(SEGMENTERS)}",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Kullanılacak görüntü sayısı."
    )
    parser.add_argument(
        "--out", default="results.csv", help="Görüntü başına metriklerin yazılacağı CSV."
    )
    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="HD95/ASSD hesaplamayı atla (daha hızlı).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Yöntemler yerine ön işleme ablasyonunu çalıştır.",
    )
    parser.add_argument(
        "--ablation-method",
        default="otsu",
        help="Ablasyonda kullanılacak segmentasyon yöntemi.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Kaç görüntüde bir ilerleme yazdırılsın.",
    )
    return parser.parse_args(argv)


def run_evaluation(dataset, pipelines, with_boundary=True, progress_every=25):
    """Her örnek için her pipeline'ı çalıştırıp uzun formatlı sonuç üretir."""
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()

    for index, sample in enumerate(dataset, start=1):
        try:
            image, truth = sample.load()
        except (FileNotFoundError, ValueError) as error:
            failures.append((sample.name, str(error)))
            continue

        for name, pipeline in pipelines.items():
            try:
                prediction = align(pipeline.predict(image), truth)
                metrics = compute_metrics(prediction, truth, with_boundary=with_boundary)
            except Exception as error:  # noqa: BLE001 - tek görüntü tüm taramayı düşürmesin
                failures.append((f"{sample.name}/{name}", str(error)))
                continue

            rows.append(
                {
                    "image": sample.name,
                    "method": name,
                    "lesion_ratio": float(truth.mean()),
                    **metrics,
                }
            )

        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = index / elapsed
            print(
                f"  {index}/{len(dataset)} goruntu | {rate:.1f} gor/s | "
                f"gecen {elapsed:.0f}s",
                flush=True,
            )

    return pd.DataFrame(rows), failures


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Yöntem başına ortalama/std/medyan özeti üretir."""
    available = [m for m in METRIC_NAMES if m in results.columns]
    summary = results.groupby("method")[available].agg(["mean", "std", "median"])
    # Dice ortalamasına göre en iyiden kötüye sırala.
    return summary.sort_values(("dice", "mean"), ascending=False)


def print_summary(results: pd.DataFrame) -> None:
    summary = summarize(results)

    print("\n=== Ozet (yontem bazinda ortalama) ===")
    header_metrics = [m for m in METRIC_NAMES if m in results.columns]
    header = f"{'yontem':<12}" + "".join(f"{m:>13}" for m in header_metrics)
    print(header)
    print("-" * len(header))
    for method in summary.index:
        line = f"{method:<12}"
        for metric in header_metrics:
            mean = summary.loc[method, (metric, "mean")]
            line += f"{mean:>13.4f}"
        print(line)

    print("\n(std) " + " ".join(f"{m}={summary.iloc[0][(m, 'std')]:.4f}" for m in header_metrics[:4]))
    print(f"En iyi (Dice ortalamasi): {summary.index[0]}")
    print("Not: " + ", ".join(sorted(LOWER_IS_BETTER & set(header_metrics))) + " icin kucuk deger iyidir.")


def paired_comparison(results: pd.DataFrame, metric: str = "dice") -> pd.DataFrame | None:
    """Yöntemleri aynı görüntüler üzerinde eşleştirilmiş testle karşılaştırır.

    Ortalamalara bakmak tek başına yetersiz: iki yöntemin ortalaması
    birbirine yakın olabilir ama biri sistematik olarak daha iyi olabilir.
    Wilcoxon işaretli sıra testi normallik varsayımı gerektirmediği için
    Dice dağılımları çarpık olduğunda t-testinden daha uygun.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("\nscipy bulunamadi, istatistiksel karsilastirma atlandi.")
        return None

    pivot = results.pivot_table(index="image", columns="method", values=metric)
    pivot = pivot.dropna()
    methods = list(pivot.columns)
    if len(pivot) < 10 or len(methods) < 2:
        print("\nEslestirilmis test icin yeterli ortak goruntu yok, atlandi.")
        return None

    rows = []
    for i, first in enumerate(methods):
        for second in methods[i + 1 :]:
            a, b = pivot[first].to_numpy(), pivot[second].to_numpy()
            difference = a - b
            if np.allclose(difference, 0):
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = wilcoxon(a, b)
            rows.append(
                {
                    "A": first,
                    "B": second,
                    f"{metric}_A": a.mean(),
                    f"{metric}_B": b.mean(),
                    "fark": difference.mean(),
                    "A_kazandi_%": 100.0 * float((difference > 0).mean()),
                    "p": p_value,
                    "anlamli_p<0.05": bool(p_value < 0.05),
                }
            )

    table = pd.DataFrame(rows)
    print(f"\n=== Eslestirilmis karsilastirma ({metric}, n={len(pivot)} goruntu) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        dataset = LesionDataset(args.data, limit=args.limit)
    except DatasetNotFound as error:
        print(f"HATA: {error}", file=sys.stderr)
        print(
            "\nVeri seti olmadan deneme icin: python scripts/demo_synthetic.py",
            file=sys.stderr,
        )
        return 1

    if len(dataset) == 0:
        print("HATA: Eslesen goruntu-maske cifti bulunamadi.", file=sys.stderr)
        return 1

    if args.ablation:
        pipelines = ablation_pipelines(args.ablation_method)
        print(f"Ablasyon modu (yontem={args.ablation_method})")
    else:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in requested if m not in SEGMENTERS]
        if unknown:
            print(
                f"HATA: Bilinmeyen yontem(ler): {', '.join(unknown)}. "
                f"Secenekler: {', '.join(SEGMENTERS)}",
                file=sys.stderr,
            )
            return 1
        pipelines = build_pipelines(requested)

    print(dataset)
    print(f"{len(pipelines)} yapilandirma x {len(dataset)} goruntu degerlendiriliyor...")

    results, failures = run_evaluation(
        dataset,
        pipelines,
        with_boundary=not args.no_boundary,
        progress_every=args.progress_every,
    )

    if results.empty:
        print("HATA: Hic sonuc uretilemedi.", file=sys.stderr)
        for name, message in failures[:10]:
            print(f"  {name}: {message}", file=sys.stderr)
        return 1

    output_path = Path(args.out)
    results.to_csv(output_path, index=False)
    print(f"\nGoruntu basina sonuclar yazildi: {output_path.resolve()}")

    print_summary(results)
    paired_comparison(results)

    if failures:
        print(f"\n{len(failures)} basarisiz oge (ilk 5):")
        for name, message in failures[:5]:
            print(f"  {name}: {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
