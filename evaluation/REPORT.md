# Valutazione detector — Grounding DINO vs OmDet-Turbo

Confronto quantitativo dei due detector open-vocabulary del PPE Tracker, su
**accuratezza statica** (dataset etichettato) e **comportamento temporale** (video).
Le soglie usate sono **identiche a quelle di produzione** (video): soglia base del
backend + soglie per-classe `DETECTION_CONF` (Glasses 0.45, Cigarette 0.50).

Data: 2026-07-24 · Hardware: CPU-only (Intel Iris Xe, 16 GB) · IoU match = 0.5

---

## 1. Verdetto

- **Grounding DINO è globalmente più accurato** (macro-F1 **0.576** vs 0.508; precisione più alta su quasi tutte le classi), ma **~6-9× più lento**.
- **OmDet-Turbo è l'unico praticabile in near-real-time su CPU** (5.0 FPS vs 0.6 FPS), a costo di più falsi positivi (precisione più bassa, specie Vest e Shoe).

**Raccomandazione operativa:**
- **Analisi offline / massima accuratezza / GPU** → **Grounding DINO**.
- **Sorveglianza live su CPU / throughput** → **OmDet-Turbo**, tenendo la sliding-window + i check geometrici a valle per contenere i falsi positivi.

---

## 2. Setup

### Dataset — SH17
Il dataset PPE più completo trovato pubblicamente: **[fathansanum/SH-17-Dataset](https://huggingface.co/datasets/fathansanum/SH-17-Dataset)** (HuggingFace, export Roboflow COCO, **CC BY 4.0**, 8099 immagini 640×640, annotazioni **complete** — non parziali come il "PPE Combined Model" usato in `benchmark_data/`).
Copre **6 delle nostre 7 classi** (manca solo *Cigarette*, assente in ogni dataset PPE).

**Mappatura classi verificata VISIVAMENTE** (l'export Roboflow ha rinominato le classi con indici numerici NON allineati all'ordine ufficiale SH17 → mappa ricavata ispezionando i crop, non indovinata):

| `category_id` COCO | Contenuto reale | Nostra classe |
|---|---|---|
| 1 | persona | Person |
| 3 | casco | Helmet |
| 9 | gilet alta visibilità | Vest |
| 16 | occhiali | Glasses |
| 17 | mani guantate | Glove |
| 7 | scarpe/stivali calzati | Shoe |

### Subset di valutazione (bilanciato)
259 immagini, campionate con quote per-classe (seed 42) per bilanciare le classi PPE rare:

| Classe | Immagini | Istanze (GT) |
|---|---|---|
| Person | 248 | 705 |
| Shoe | 113 | 414 |
| Vest | 112 | 261 |
| Glove | 122 | 256 |
| Helmet | 108 | 228 |
| Glasses | 100 | 121 |

---

## 3. Valutazione statica (259 immagini, IoU ≥ 0.5)

### Grounding DINO — macro-F1 **0.576** · micro-F1 **0.63**

| Classe | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Person | 0.94 | 0.72 | **0.81** | 507 | 32 | 198 |
| Helmet | 0.69 | 0.58 | **0.63** | 132 | 58 | 96 |
| Glasses | 0.67 | 0.46 | **0.55** | 56 | 28 | 65 |
| Vest | 0.59 | 0.45 | **0.51** | 117 | 81 | 144 |
| Glove | 0.64 | 0.43 | **0.51** | 110 | 63 | 146 |
| Shoe | 0.80 | 0.30 | **0.44** | 126 | 31 | 288 |

### OmDet-Turbo — macro-F1 **0.508** · micro-F1 **0.558**

| Classe | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| Person | 0.88 | 0.70 | **0.78** | 497 | 70 | 208 |
| Helmet | 0.54 | 0.65 | **0.59** | 149 | 129 | 79 |
| Glove | 0.54 | 0.40 | **0.46** | 102 | 87 | 154 |
| Glasses | 0.73 | 0.31 | **0.44** | 38 | 14 | 83 |
| Shoe | 0.39 | 0.46 | **0.42** | 192 | 307 | 222 |
| Vest | 0.34 | 0.39 | **0.36** | 102 | 200 | 159 |

### Lettura
- **Person**: entrambi affidabili (F1 ~0.8). Recall ~0.7 perché SH17 annota anche persone molto piccole/parziali sullo sfondo che il detector a soglia 0.35 non prende.
- **Helmet/Vest**: GD nettamente più preciso (Vest P 0.59 vs 0.34; Helmet P 0.69 vs 0.54). OmDet recupera qualche recall in più ma con molti FP (Vest 200 FP, Shoe 307 FP).
- **Shoe**: recall basso per entrambi (GD 0.30) — scarpe piccole, spesso occluse/parziali; GD compensa con precisione alta (0.80).
- **Glasses**: buona precisione (0.67–0.73), recall basso — c'è un *domain gap* (SH17 annota occhiali generici/da vista, il prompt è "safety glasses").
- **Cigarette (proxy FP)**: SH17 non annota sigarette → non calcolabile P/R. Detection "Cigarette" spurie a livello grezzo: **GD 18, OmDet 4**. ⚠️ Sono conteggi *detection-level*: in questo test **non** sono applicati i check geometrici di plausibilità (richiedono l'associazione alla persona); nel pipeline video quei check + la persistenza li avevano azzerati.

---

## 4. Statistiche temporali (video `ppe_video.mp4`, 240 frame 1280×720, skip=8)

| Metrica | Grounding DINO | OmDet-Turbo |
|---|---|---|
| **FPS effettivi (end-to-end)** | 0.59 | **5.05** |
| Latenza detection — mediana | 12.6 s | **1.37 s** |
| Latenza detection — p95 | 13.5 s | **1.44 s** |
| Wall time (10 s di video) | 407 s | **47.5 s** |
| Persone/frame (media) | 1.83 | 1.87 |
| **Track creati (stabilità identità)** | **3** | 5 |
| Alert confermati | 13 | 20 |
| Tempo al 1° alert | 0.08 s | 0.08 s |

> La latenza *mediana* è la misura rappresentativa; il `max` (39 s GD / 6.3 s OmDet) include il caricamento del modello alla prima inferenza.

### Lettura
- **Throughput**: OmDet **~8.6× più veloce** end-to-end. Su CPU, OmDet (~5 FPS) è al limite dell'usabilità live; Grounding DINO (0.6 FPS) è di fatto solo per analisi offline/batch o GPU.
- **Stabilità identità**: GD crea **3** track per ~2 persone in scena, OmDet **5** → GD ha meno "ID switch", tracking più stabile (coerente con la sua maggior precisione delle bbox persona).
- **Sensibilità agli alert**: OmDet conferma più alert (20 vs 13), coerente con la precisione più bassa → più trigger, anche spuri.

---

## 5. Limiti e note metodologiche (onestà sui numeri)

- **Metrica al punto operativo, non mAP**: P/R/F1 sono misurati alle soglie di *produzione* (quelle dei video). Riflettono la resa del sistema così com'è configurato, non la qualità intrinseca a soglia variabile (mAP). Un confronto mAP@[.5:.95] richiederebbe di rieseguire a soglia bassa (raddoppiando i tempi di GD).
- **Check geometrici non applicati nella statica**: i filtri di plausibilità (occhiali in fascia-volto, dimensione/larghezza sigaretta) sono logica di *sistema* a valle (richiedono l'associazione persona↔oggetto) e non sono attivi in questo test detection-level. Quindi i FP statici (specie Cigarette) sovrastimano quelli del pipeline completo.
- **Domain gap classi**: "safety glasses"/"work glove"/"work boot" (prompt) vs occhiali/guanti/scarpe generici (annotazioni SH17) penalizzano il recall — un limite dello zero-shot, non necessariamente del modello.
- **Recall Person < 1**: SH17 etichetta persone minuscole/di sfondo che la soglia 0.35 scarta; non è un errore grave per il caso d'uso (sicurezza sugli operai in primo piano).
- **Completezza annotazioni**: SH17 è densamente annotato, ma eventuali oggetti non etichettati gonfierebbero i FP. Il **confronto relativo** tra i due detector resta valido (stessa GT).
- **Cigarette non valutabile staticamente**: nessun dataset PPE la annota; la sua validazione resta quella qualitativa sui video.

---

## 6. File prodotti

```
evaluation/
├── REPORT.md                      # questo report
├── build_subset.py                # costruzione subset bilanciato SH17 (riproducibile, seed 42)
├── eval_static.py                 # valutazione statica P/R/F1
├── temporal_stats.py              # statistiche temporali sui video
├── eval_static_results.json       # risultati statici (per-classe, macro/micro, latenza)
├── temporal_stats_results.json    # risultati temporali (FPS, latenza, track, alert)
├── sh17_subset/
│   ├── images/                    # 259 immagini del subset (SH17, CC BY 4.0)
│   └── ground_truth.json          # GT rimappata sulle nostre 6 classi
├── sh17_meta/                     # cache download annotazioni COCO (eliminabile)
└── verify/                        # crop usati per verificare la mappatura classi
```

### Riproducibilità
```bash
python evaluation/build_subset.py      # ricostruisce il subset (serve HF_TOKEN nel .env)
python evaluation/eval_static.py       # valutazione statica (GD ~40min, OmDet ~6min su CPU)
python evaluation/temporal_stats.py    # statistiche temporali sui video
```

**Attribuzione dataset**: SH17 — https://universe.roboflow.com/safety-measure/sh17-dataset — licenza CC BY 4.0.
