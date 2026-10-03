"""README için figür üretir.

En iyi ve en kötü sonuç veren örnekleri bulup tek bir panelde gösterir:
girdi → ön işleme → gerçek maske → dört yöntemin tahmini.

Kullanım:
    python scripts/make_figures.py --data data --limit 300
    python scripts/make_figures.py --results results.csv --data data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Ekran olmayan ortamda da kaydedebilsin

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DatasetNotFound, LesionDataset  # noqa: E402
from src.pipeline import build_pipelines  # noqa: E402
from src import viz  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Veri seti kökü.")
    parser.add_argument(
        "--results",
        default=None,
        help="Varsa evaluate.py çıktısı; en iyi/kötü örnekler buradan seçilir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="--results verilmediğinde taranacak görüntü sayısı.",
    )
    parser.add_argument("--out", default="docs", help="Figürlerin yazılacağı klasör.")
    parser.add_argument("--metric", default="dice", help="Seçim ve etiket metriği.")
    return parser.parse_args(argv)


def rank_images(dataset, pipelines, metric: str) -> pd.Series:
    """Her görüntü için yöntemlerin ortalama metriğini hesaplar."""
    scores = {}
    for index, sample in enumerate(dataset, start=1):
        image, truth = sample.load()
        values = [
            pipeline.evaluate(image, truth, with_boundary=False)[metric]
            for pipeline in pipelines.values()
        ]
        scores[sample.name] = float(np.nanmean(values))
        if index % 25 == 0:
            print(f"  {index}/{len(dataset)} taranan", flush=True)
    return pd.Series(scores).sort_values()


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        dataset = LesionDataset(args.data, limit=None if args.results else args.limit)
    except DatasetNotFound as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 1

    pipelines = build_pipelines()

    if args.results and Path(args.results).exists():
        results = pd.read_csv(args.results)
        ranking = results.groupby("image")[args.metric].mean().sort_values()
        print(f"Siralama {args.results} dosyasindan okundu ({len(ranking)} goruntu).")
    else:
        print(f"{len(dataset)} goruntu taraniyor...")
        ranking = rank_images(dataset, pipelines, args.metric)

    lookup = {sample.name: sample for sample in LesionDataset(args.data)}
    worst_name, best_name = str(ranking.index[0]), str(ranking.index[-1])

    cases = []
    for label, name in (("Başarılı örnek", best_name), ("Zor örnek", worst_name)):
        sample = lookup.get(name)
        if sample is None:
            print(f"UYARI: {name} veri setinde bulunamadi, atlandi.", file=sys.stderr)
            continue
        image, truth = sample.load()
        cases.append((f"{label}\n{name}", image, truth))

    if not cases:
        print("HATA: Gosterilecek ornek bulunamadi.", file=sys.stderr)
        return 1

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = viz.showcase(cases, pipelines, metric=args.metric)
    overview_path = output_dir / "overview.png"
    figure.savefig(overview_path, dpi=120, bbox_inches="tight")
    print(f"Yazildi: {overview_path.resolve()}")

    # Ön işleme adımlarını ayrı bir figürde göster.
    image, _ = lookup[best_name].load()
    stages = next(iter(pipelines.values())).run(image)
    stage_figure = viz.show_stages(stages, title="Ön işleme zinciri")
    stages_path = output_dir / "preprocessing.png"
    stage_figure.savefig(stages_path, dpi=120, bbox_inches="tight")
    print(f"Yazildi: {stages_path.resolve()}")

    print(
        f"\nSecilen ornekler:\n  en iyi : {best_name} ({args.metric}={ranking.iloc[-1]:.3f})"
        f"\n  en kotu: {worst_name} ({args.metric}={ranking.iloc[0]:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
