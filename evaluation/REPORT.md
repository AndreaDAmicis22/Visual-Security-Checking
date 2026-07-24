# Valutazione detector — Grounding DINO vs OmDet-Turbo

Confronto quantitativo dei due detector open-vocabulary del PPE Tracker, su
**accuratezza statica** (dataset etichettato) e **comportamento temporale** (video).
Le soglie usate sono **identiche a quelle di produzione** (video): soglia base del
backend + soglie per-classe `DETECTION_CONF` (Glasses 0.45, Cigarette 0.50).

Data: 2026-07-24 · Hardware: CPU-only (Intel Iris Xe, 16 GB) · IoU match = 0.5

---

## 1. Verdetto

- **Grounding DINO è globalmente più accurato** (mAP@.5 **0.516** vs 0.482; precisione più alta su Person/Vest/Glasses), ma **~6-9× più lento**.
- **OmDet-Turbo è l'unico praticabile in near-real-time su CPU** (5.0 FPS vs 0.6 FPS), leggermente migliore su Helmet/Shoe ma con più falsi positivi (precisione più bassa, specie Vest).
- **Le metriche sono modeste ma oneste**: il cuore del sistema (rilevare gli operai) è solido (Person AP@.5 ~0.80); il calo è su guanti/scarpe/gilet/occhiali — oggetti piccoli, difficili in **zero-shot senza training**. Non è un artefatto delle soglie (ottimizzarle dà solo +2 punti F1). Vedi §3 per la scomposizione.

**Raccomandazione operativa:**
- **Analisi offline / massima accuratezza / GPU** → **Grounding DINO**.
- **Sorveglianza live su CPU / throughput** → **OmDet-Turbo**, tenendo la sliding-window + i check geometrici a valle per contenere i falsi positivi.
- **Per migliorare davvero le PPE-class piccole** servirebbe un fine-tuning leggero (o soglie/IoU tarati per classe): lo zero-shot puro ha un tetto su guanti/scarpe.

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

## 3. Valutazione statica (259 immagini)

Metriche calcolate su detection grezze catturate a soglia bassa (`eval_capture.py`)
e ricalcolate offline (`eval_metrics.py`), così da riportare metriche indipendenti
dalla soglia (**AP/mAP**) accanto al punto operativo:
- **AP@.5 / AP@.3**: average precision (area sotto la curva P-R), a IoU 0.5 e 0.3.
  Indipendente dalla soglia → qualità "intrinseca" del detector.
- **F1 prod**: F1 al punto operativo di *produzione* (soglie dei video: base 0.35 GD /
  0.30 OmDet, Glasses 0.45).
- **F1 best**: F1 massimo ottenibile variando la sola soglia (+ soglia che lo realizza).

### Grounding DINO — **mAP@.5 = 0.516** · mAP@.3 = 0.576 · macro-F1(prod) = 0.547

| Classe | AP@.5 | AP@.3 | F1 prod (P/R) | F1 best @soglia (P/R) |
|---|---|---|---|---|
| Person | 0.80 | 0.83 | **0.81** (0.94/0.72) | 0.83 @0.30 (0.89/0.77) |
| Helmet | 0.59 | 0.60 | **0.62** (0.66/0.58) | 0.62 @0.35 (0.66/0.58) |
| Glasses | 0.50 | 0.61 | **0.54** (0.67/0.46) | 0.57 @0.35 (0.57/0.57) |
| Vest | 0.48 | 0.49 | **0.50** (0.58/0.44) | 0.52 @0.30 (0.54/0.50) |
| Glove | 0.39 | 0.48 | **0.45** (0.50/0.41) | 0.45 @0.35 (0.50/0.41) |
| Shoe | 0.34 | 0.45 | **0.35** (0.77/0.23) | 0.43 @0.25 (0.54/0.36) |

### OmDet-Turbo — **mAP@.5 = 0.482** · mAP@.3 = 0.555 · macro-F1(prod) = 0.508

| Classe | AP@.5 | AP@.3 | F1 prod (P/R) | F1 best @soglia (P/R) |
|---|---|---|---|---|
| Person | 0.78 | 0.81 | **0.78** (0.88/0.70) | 0.78 @0.25 (0.83/0.74) |
| Helmet | 0.61 | 0.65 | **0.59** (0.54/0.65) | 0.61 @0.50 (0.67/0.56) |
| Glasses | 0.46 | 0.59 | **0.44** (0.73/0.31) | 0.51 @0.35 (0.58/0.45) |
| Glove | 0.39 | 0.45 | **0.46** (0.54/0.40) | 0.46 @0.30 (0.54/0.40) |
| Shoe | 0.39 | 0.50 | **0.42** (0.39/0.46) | 0.46 @0.40 (0.56/0.40) |
| Vest | 0.26 | 0.34 | **0.36** (0.34/0.39) | 0.36 @0.30 (0.34/0.39) |

### Lettura (onesta) — perché i numeri sono modesti
Tre fattori, quantificati:

1. **Difficoltà intrinseca dello zero-shot su oggetti piccoli** — è la causa *principale*.
   Person è solido (AP@.5 ~0.80: il cuore del sistema — trovare gli operai — funziona),
   ma guanti/scarpe/gilet (piccoli, in coppia, spesso parziali) stanno a AP 0.26–0.48.
   Nessun training PPE → limite atteso.
2. **Severità dell'IoU** — moderata. mAP@.3 è ~6 punti sopra mAP@.5 (GD 0.576 vs 0.516):
   molti box sono "quasi giusti" ma non centrano l'IoU 0.5, specie sugli oggetti piccoli
   (Shoe AP .34→.45, Glasses .50→.61). Per una detection di sicurezza grossolana, IoU 0.3
   è più rappresentativo.
3. **Soglia di produzione** — impatto *piccolo*, contrariamente a quanto ipotizzato all'inizio:
   il macro-F1 al punto operativo (0.547) è a solo ~2 punti dal massimo ottenibile ottimizzando
   la soglia (0.57). L'unica classe che ne soffre davvero è **Shoe** (F1 0.35→0.43 abbassando
   la soglia a 0.25). Quindi le soglie alte tarate sui video **non** stanno "rovinando" il recall
   in modo drammatico: i numeri sono quelli reali dello zero-shot.

**Confronto detector**: GD ha mAP più alto su Person, Vest, Glasses; OmDet è leggermente
migliore su Helmet e Shoe. Nel complesso **GD > OmDet** (mAP@.5 0.516 vs 0.482).

- **Cigarette (proxy FP)**: SH17 non annota sigarette → non calcolabile P/R. Detection grezze
  a soglia 0.05: **GD 1302, OmDet 2832** su 259 immagini; alla soglia di produzione 0.50 scendono
  a **GD 15, OmDet 4**. ⚠️ Conteggi *detection-level*: i check geometrici di plausibilità (che
  nei video avevano azzerato i FP) **non** sono applicati qui → sovrastimano i FP del pipeline reale.

### Effetto della NMS per-classe (prova)
Aggiunta una **Non-Maximum Suppression per-classe** (IoU 0.5, `nms_per_class` in `analyzer.py`)
per rimuovere i box duplicati/sovrapposti della stessa classe. Ricalcolo offline sui dump
(variante `nms` in `eval_metrics_results.json`):

| | mAP@.5 (senza→con NMS) | macro-F1 prod |
|---|---|---|
| Grounding DINO | 0.516 → **0.519** (+0.003) | 0.547 → 0.548 |
| OmDet-Turbo | 0.482 → **0.482** (+0.000) | 0.508 → 0.508 |

**Esito onesto: sulle metriche l'effetto è trascurabile.** Motivo: (a) la valutazione mAP già
"assorbe" i duplicati (i box extra sullo stesso GT diventano FP a bassa confidence, poco pesanti
sull'AP); (b) le over-detection sono per lo più box *spazialmente distinti*, non duplicati
sovrapposti (la NMS rimuove solo 3 box su OmDet, 154 su GD). La NMS **resta utile** per: pulire
l'output visivo (meno box accavallati nei video) e ridurre qualche FP; **non** è la leva per
alzare le metriche — quelle sono limitate dalla difficoltà zero-shot sugli oggetti piccoli.

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

- **AP a IoU singolo, non COCO mAP@[.5:.95]**: si riportano AP@.5 e AP@.3 (più informativi per detection di sicurezza grossolana) ma non la media COCO su 10 soglie IoU; sarebbe più severa e non aggiungerebbe molto al confronto relativo tra i due modelli.
- **Check geometrici non applicati nella statica**: i filtri di plausibilità (occhiali in fascia-volto, dimensione/larghezza sigaretta) sono logica di *sistema* a valle (richiedono l'associazione persona↔oggetto) e non sono attivi in questo test detection-level. Quindi i FP statici (specie Cigarette) sovrastimano quelli del pipeline completo.
- **Domain gap classi**: "safety glasses"/"work glove"/"work boot" (prompt) vs occhiali/guanti/scarpe generici (annotazioni SH17) penalizzano il recall — un limite dello zero-shot, non necessariamente del modello.
- **Recall Person < 1**: SH17 etichetta persone minuscole/di sfondo che la soglia 0.35 scarta; non è un errore grave per il caso d'uso (sicurezza sugli operai in primo piano).
- **Completezza annotazioni**: SH17 è densamente annotato, ma eventuali oggetti non etichettati gonfierebbero i FP. Il **confronto relativo** tra i due detector resta valido (stessa GT).
- **Cigarette non valutabile staticamente**: nessun dataset PPE la annota; la sua validazione resta quella qualitativa sui video.

---

## 6. File prodotti

```
evaluation/
├── REPORT.md                        # questo report
├── build_subset.py                  # costruzione subset bilanciato SH17 (riproducibile, seed 42)
├── eval_static.py                   # valutazione statica P/R/F1 al punto operativo (soglie di produzione)
├── eval_capture.py                  # cattura detection grezze (soglia bassa, filtro per-classe off)
├── eval_metrics.py                  # metriche offline dal dump: mAP@.5/.3, F1 prod vs best, sweep soglia
├── temporal_stats.py                # statistiche temporali sui video
├── eval_static_results.json         # risultati statici al punto operativo
├── eval_metrics_results.json        # mAP + F1 prod/best per-classe (analisi principale)
├── temporal_stats_results.json      # risultati temporali (FPS, latenza, track, alert)
├── raw_detections_*.json            # dump detection grezze per detector (per ri-analisi offline)
├── sh17_subset/
│   ├── images/                      # 259 immagini del subset (SH17, CC BY 4.0) [gitignored]
│   └── ground_truth.json            # GT rimappata sulle nostre 6 classi
├── sh17_meta/                       # cache download annotazioni COCO (eliminabile) [gitignored]
└── verify/                          # crop verifica classi + montaggio predizioni [gitignored]
```

### Riproducibilità
```bash
python evaluation/build_subset.py      # ricostruisce il subset (serve HF_TOKEN nel .env)
python evaluation/eval_capture.py      # cattura detection grezze (GD ~40min, OmDet ~6min su CPU)
python evaluation/eval_metrics.py      # metriche offline dal dump (istantaneo)
python evaluation/eval_static.py       # (opz.) P/R/F1 diretto al punto operativo
python evaluation/temporal_stats.py    # statistiche temporali sui video
```

Il montaggio qualitativo **GT vs Grounding DINO vs OmDet** è in `evaluation/verify/predictions_montage.jpg`.

**Attribuzione dataset**: SH17 — https://universe.roboflow.com/safety-measure/sh17-dataset — licenza CC BY 4.0.
