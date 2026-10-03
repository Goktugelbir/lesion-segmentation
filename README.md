# Deri Lezyonu Segmentasyonu — Klasik Görüntü İşleme

Dermoskopi görüntülerinde lezyon sınırını **derin öğrenme kullanmadan** bulan bir
pipeline. Dört segmentasyon yöntemi, ortak ön/son işleme zinciri üzerinde adil biçimde
karşılaştırılıyor ve sonuçlar istatistiksel testle doğrulanıyor.

## Neden klasik yöntemler?

Etiketli veri gerektirmez, saniyeler içinde çalışır ve her kararı açıklanabilir —
hangi eşiğin neden seçildiği görülebilir. Buradaki asıl soru şu: *dermoskopi
artefaktları düzgün temizlendiğinde klasik eşikleme nereye kadar gider?*

## Pipeline

```
Görüntü
  ├─ Ön işleme
  │    ├─ DullRazor kıl temizleme   (4 yönlü doğrusal kapama + inpaint)
  │    ├─ Median filtre             (kalan nokta gürültüsü)
  │    ├─ Gölge düzeltme            (deri piksellerine kuadratik yüzey fit'i)
  │    └─ CLAHE                     (LAB uzayında yalnızca L kanalı)
  ├─ Segmentasyon                   (otsu | adaptive | kmeans | watershed)
  └─ Son işleme
       ├─ Açma → küçük bölge eleme
       ├─ Delik doldurma
       ├─ Kapama
       └─ Bölge seçimi              (merkez yakınlığı + kenar cezası)
```

### Ön işlemedeki üç tasarım kararı

**Gölge düzeltme neden bulanıklaştırma ile yapılmıyor?**
Yaygın yaklaşım, aydınlatma alanını görüntünün çok bulanık halinden tahmin etmektir.
Büyük bir lezyon da koyu olduğu için bu tahmine karışır; bölme işlemi lezyonu
aydınlatır ve segmentasyon için gereken kontrast kaybolur. Bunun yerine parlaklık
kanalına **yalnızca deri piksellerinden** (parlaklığı medyanın üzerindekiler) ikinci
derece bir yüzey oturtuluyor. Vinyet merkezden uzaklıkla yaklaşık kuadratik azaldığı
için bu model yeterli. `tests/test_pipeline.py::test_shading_correction_preserves_lesion_contrast`
bu gerilemeyi yakalıyor.

**Bölge seçiminde neden kenar cezası var?**
Lezyon kadraja ortalanmış olur, bu yüzden "merkeze en yakın büyük bölgeyi seç" makul
görünür. Ama vinyetten doğan **halka şeklindeki** bir artefaktın ağırlık merkezi de tam
görüntü merkezindedir ve alanı daha büyüktür — yalnızca merkez mesafesine bakan bir
seçim halkayı lezyon sanar. Bu yüzden görüntü çerçevesini dolaşan bölgeler
cezalandırılıyor.

**Delik doldurma neden bölge seçiminden önce?**
Adaptif eşikleme geniş ve düz bir lezyonun yalnızca kenarını yakalar (iç bölgede yerel
kontrast yoktur). Bu ince halka doldurulmazsa alanı küçük kalır ve bölge seçiminde
gürültüye kaybeder.

## Segmentasyon yöntemleri

| Yöntem | Fikir | Güçlü yanı | Zayıf yanı |
|--------|-------|-----------|-----------|
| `otsu` | Histogramı iki sınıfa ayıran tek global eşik | Hızlı, parametresiz | Lezyon–deri geçişi yumuşaksa eşiği kaydırır |
| `adaptive` | Eşik her piksel için komşuluğundan | Düzgün olmayan aydınlatmaya dayanıklı | Geniş lezyonun içini kaçırır; delik doldurmaya bağımlı |
| `kmeans` | LAB uzayında renk kümeleme, en koyu küme | Parlaklık farkı az, renk farkı belirginse iyi | Başlangıca duyarlı, daha yavaş |
| `watershed` | Otsu'dan işaretleyici, gradyana göre havza | Sınırı gerçek kenara oturtur | Yanlış işaretleyici → aşırı/eksik bölütleme |

Dördü de **aynı** ön ve son işlemeyi kullanır; aradaki tek fark eşik kararıdır.
Karşılaştırmanın anlamlı olması için bu şart.

## Metrikler

| Metrik | Ne ölçer | Not |
|--------|----------|-----|
| IoU, Dice | Bölge örtüşmesi | Standart; sınır kalitesine duyarsız |
| Sensitivity, Specificity, Precision | Hata tipi dağılımı | Yöntem aşırı mı yoksa eksik mi bölütlüyor |
| Accuracy | Piksel doğruluğu | **Tek başına yanıltıcı:** lezyon %10 ise her şeye "deri" diyen bir yöntem %90 alır |
| XOR hatası | `(FP+FN) / lezyon alanı` | Küçük lezyonlarda accuracy'den çok daha ayırt edici |
| HD95, ASSD | Sınır mesafesi (piksel) | Aynı Dice'lı iki maske farklı sınır kalitesinde olabilir. **Küçük = iyi** |

Yöntemler ayrıca **eşleştirilmiş Wilcoxon işaretli sıra testi** ile karşılaştırılıyor:
iki ortalamanın yakın olması "fark yok" demek değildir, bir yöntem sistematik olarak
az farkla ama tutarlı biçimde daha iyi olabilir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Veri seti olmadan (kodu denemek için)

```bash
python scripts/demo_synthetic.py
```

Sentetik dermoskopi görüntüleri (koyu lezyon + kıllar + vinyet) üretir ve dört yöntemi
karşılaştırır. **Bu görüntüler gerçek dermoskopiden belirgin olarak kolaydır** — çıkan
skorlar kodun çalıştığını gösterir, gerçek performansı değil.

### Gerçek veri seti ile

Veri seti: [skin-cancer-lesions-segmentation](https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation)

Beklenen yapı (klasör adları `images`/`masks`, `image`/`mask`, `gt` … olabilir;
otomatik bulunur):

```
data/
├── images/   ISIC_0024306.jpg
└── masks/    ISIC_0024306.png
```

```bash
# Tüm yöntemler, 300 görüntü
python scripts/evaluate.py --data data --limit 300

# Sadece iki yöntem, sınır metrikleri olmadan (daha hızlı)
python scripts/evaluate.py --methods otsu,kmeans --no-boundary

# Ön işleme ablasyonu: hangi adım ne kadar katkı veriyor
python scripts/evaluate.py --ablation --limit 200
```

Çıktı: görüntü başına metrikleri içeren `results.csv` + konsola özet tablo ve
eşleştirilmiş karşılaştırma.

### Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Veri seti bulunursa onu kullanır, bulunamazsa otomatik olarak sentetik moda geçer —
her ortamda çalışır.

## Proje yapısı

```
src/
├── data.py         Veri seti keşfi, görüntü/maske yükleme
├── preprocess.py   DullRazor, gölge düzeltme, CLAHE, FOV maskesi
├── segment.py      Dört segmentasyon yöntemi + kayıt (registry)
├── postprocess.py  Temizlik, delik doldurma, bölge seçimi
├── metrics.py      IoU/Dice/XOR/HD95/ASSD
├── pipeline.py     Aşamaları birleştiren tek giriş noktası
├── viz.py          Görselleştirme
└── synthetic.py    Veri seti olmadan test için sentetik görüntü üretimi
scripts/
├── evaluate.py        Toplu değerlendirme CLI'si
└── demo_synthetic.py  Veri setsiz demo
tests/
└── test_pipeline.py   17 test
notebooks/
└── analysis.ipynb     Uçtan uca analiz
```

Yeni bir segmentasyon yöntemi eklemek için `src/segment.py` içine fonksiyonu yazıp
`SEGMENTERS` sözlüğüne bir satır eklemek yeterli; değerlendirme betiği ve notebook
otomatik olarak onu da karşılaştırmaya alır.

## Testler

```bash
python tests/test_pipeline.py        # pytest kurulu olmasa da çalışır
python -m pytest tests -q            # pytest varsa
```

Testler metriklerin elle hesaplanabilir değerlerde doğrulanmasını, ön işleme
adımlarının beklenen davranışını ve dört yöntemin uçtan uca çalışmasını kapsıyor.
Gerçek veri gerektirmezler.

## Bilinen sınırlar

Klasik eşiklemenin zorlandığı durumlar: çoklu lezyon, lezyonun kadrajı taşması,
mürekkep işaretleri, jel/balon yansımaları ve açık renkli (amelanotik) lezyonlar.
Pipeline'ın temel varsayımı **lezyonun çevresindeki deriden koyu olmasıdır**; bu
varsayım bozulduğunda dört yöntem de başarısız olur. Bu senaryolar için öğrenme
tabanlı yöntemler gerekir.
