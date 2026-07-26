# Valutazione dei detector e ricerca del miglior setup — PPE Tracker

Questo report racconta, in ordine, **come abbiamo valutato i detector open-vocabulary** del PPE
Tracker e **tutti i tentativi fatti per migliorare le metriche**, con il *perché* di ogni scelta e
il relativo *esito*. Si parte dal confronto Grounding DINO vs OmDet-Turbo, si diagnostica perché i
numeri sono modesti, e si prova una serie di soluzioni fino a trovare la configurazione migliore.

Data: 2026-07-24 · Hardware: **CPU-only** (Intel Iris Xe, 16 GB) · Vincoli: **licenza Apache 2.0**
(no Ultralytics/AGPL), **zero-shot** (nessun training PPE disponibile).

---

## Sommario (TL;DR)

- Punto di partenza: **Grounding DINO** (mAP@.5 0.516) è più accurato ma lento; **OmDet-Turbo**
  (0.482) è ~8.6× più veloce → unico usabile live su CPU.
- Le metriche sono modeste **per una ragione reale**: lo zero-shot fatica sugli oggetti PPE piccoli
  (guanti/scarpe/occhiali). **Non** è colpa delle soglie né dei box duplicati (verificato).
- Cosa **non** ha funzionato: prompt multi-sinonimo (marginale), SAHI/tiling (peggiora), NMS
  (irrilevante per le metriche, utile solo visivamente).
- Cosa **ha** funzionato: **cambiare detector**. **OWLv2** (Apache 2.0) è più accurato di entrambi
  (mAP@.5 **0.583**); l'**ensemble OWLv2 + Grounding DINO sugli occhiali** è il migliore in assoluto,
  **mAP@.5 0.629 (+21% sul GD baseline)** e 0 falsi positivi sui video.
- Il prossimo salto reale (specie occhiali/oggetti piccoli) sarebbe il **fine-tuning**, non
  praticabile su CPU → rimandato a GPU.

**Raccomandazione**: live su CPU → **OmDet-Turbo**; massima accuratezza offline → **ensemble
OWLv2+GD**; per un salto ulteriore → fine-tuning su GPU.

---

## 1. La domanda di partenza

Il tracker può usare due detector (`--detector`). Servivano due risposte:
1. **Quale dei due è meglio**, a livello globale, per la nostra pipeline?
2. **Metriche di validazione** oggettive, non solo l'impressione sui video demo.

Da qui: costruire un benchmark su dataset etichettato + misurare il comportamento temporale sui
video, e — visto che i numeri sono usciti modesti — capirne il motivo e provare a migliorarli.

---

## 2. Setup della valutazione

### Perché SH17 come dataset
Serviva un dataset PPE con **annotazioni complete** (quello già in `benchmark_data/`, il Roboflow
"PPE Combined Model", ha annotazioni *parziali* → falsi positivi fasulli). Scelto
**[SH17](https://huggingface.co/datasets/fathansanum/SH-17-Dataset)** (export Roboflow COCO,
**CC BY 4.0**, 8099 img 640×640, annotazioni dense): è il PPE dataset più completo trovato e copre
**6 delle nostre 7 classi** (manca solo *Cigarette*, assente in ogni dataset PPE).

**Mappatura classi verificata VISIVAMENTE** (l'export Roboflow ha rinominato le classi con indici
numerici NON allineati all'ordine ufficiale SH17 → ricavata ispezionando i crop, non indovinata):

| `category_id` COCO | Contenuto reale | Nostra classe |
|---|---|---|
| 1 | persona | Person |
| 3 | casco | Helmet |
| 9 | gilet alta visibilità | Vest |
| 16 | occhiali | Glasses |
| 17 | mani guantate | Glove |
| 7 | scarpe/stivali calzati | Shoe |

### Subset bilanciato (259 immagini)
Campionato con quote per-classe (seed 42) per bilanciare le classi PPE rare (`build_subset.py`):

| Classe | Immagini | Istanze (GT) |
|---|---|---|
| Person | 248 | 705 |
| Shoe | 113 | 414 |
| Vest | 112 | 261 |
| Glove | 122 | 256 |
| Helmet | 108 | 228 |
| Glasses | 100 | 121 |

### Perché due metriche
- **AP / mAP** (area sotto la curva Precision-Recall, a IoU 0.5 e 0.3): **indipendente dalla soglia**
  → misura la qualità "intrinseca" del detector, il confronto pulito.
- **F1 al punto operativo**: con le soglie reali di produzione → dice come rende il sistema *così
  com'è configurato*.

Per calcolarle senza rieseguire i modelli ad ogni analisi: **cattura una volta** tutte le detection
a soglia bassa (`eval_capture.py`), poi **ricalcolo offline** (`eval_metrics.py`).

---

## 3. Passo 1 — Baseline: Grounding DINO vs OmDet-Turbo

### Accuratezza statica (259 immagini)

**Grounding DINO — mAP@.5 = 0.516 · mAP@.3 = 0.576**

| Classe | AP@.5 | AP@.3 | F1 prod (P/R) | F1 best @soglia (P/R) |
|---|---|---|---|---|
| Person | 0.80 | 0.83 | **0.81** (0.94/0.72) | 0.83 @0.30 (0.89/0.77) |
| Helmet | 0.59 | 0.60 | **0.62** (0.66/0.58) | 0.62 @0.35 (0.66/0.58) |
| Glasses | 0.50 | 0.61 | **0.54** (0.67/0.46) | 0.57 @0.35 (0.57/0.57) |
| Vest | 0.48 | 0.49 | **0.50** (0.58/0.44) | 0.52 @0.30 (0.54/0.50) |
| Glove | 0.39 | 0.48 | **0.45** (0.50/0.41) | 0.45 @0.35 (0.50/0.41) |
| Shoe | 0.34 | 0.45 | **0.35** (0.77/0.23) | 0.43 @0.25 (0.54/0.36) |

**OmDet-Turbo — mAP@.5 = 0.482 · mAP@.3 = 0.555**

| Classe | AP@.5 | AP@.3 | F1 prod (P/R) | F1 best @soglia (P/R) |
|---|---|---|---|---|
| Person | 0.78 | 0.81 | **0.78** (0.88/0.70) | 0.78 @0.25 (0.83/0.74) |
| Helmet | 0.61 | 0.65 | **0.59** (0.54/0.65) | 0.61 @0.50 (0.67/0.56) |
| Glasses | 0.46 | 0.59 | **0.44** (0.73/0.31) | 0.51 @0.35 (0.58/0.45) |
| Glove | 0.39 | 0.45 | **0.46** (0.54/0.40) | 0.46 @0.30 (0.54/0.40) |
| Shoe | 0.39 | 0.50 | **0.42** (0.39/0.46) | 0.46 @0.40 (0.56/0.40) |
| Vest | 0.26 | 0.34 | **0.36** (0.34/0.39) | 0.36 @0.30 (0.34/0.39) |

**GD > OmDet** sull'accuratezza (mAP@.5 0.516 vs 0.482): GD meglio su Person/Vest/Glasses, OmDet un
filo meglio su Helmet/Shoe ma con più falsi positivi.

### Comportamento temporale (video `ppe_video.mp4`, 240 frame 1280×720, skip=8)

| Metrica | Grounding DINO | OmDet-Turbo |
|---|---|---|
| **FPS effettivi (end-to-end)** | 0.59 | **5.05** |
| Latenza detection — mediana | 12.6 s | **1.37 s** |
| Wall time (10 s di video) | 407 s | **47.5 s** |
| Track creati (stabilità identità) | **3** | 5 |
| Alert confermati | 13 | 20 |

Su CPU **solo OmDet è vicino al live** (~5 FPS); GD (0.6 FPS) è da analisi offline/GPU. GD ha però
un tracking più stabile (3 track vs 5 per ~2 persone in scena).

**Esito del Passo 1**: GD più accurato, OmDet molto più veloce — ma **entrambi con metriche modeste**
sulle classi PPE piccole. Da qui la domanda: *perché?*

---

## 4. Passo 2 — Diagnosi: perché le metriche sono modeste

Prima di "tarare a caso", abbiamo scomposto le cause (le metriche potevano essere depresse da soglia,
IoU severo, box duplicati, o difficoltà reale). Tre fattori, quantificati:

1. **Difficoltà intrinseca dello zero-shot su oggetti piccoli** — *la causa principale*. Person è
   solido (AP@.5 ~0.80: il cuore del sistema funziona), ma guanti/scarpe/gilet (piccoli, in coppia,
   spesso parziali) stanno a AP 0.26–0.48. Senza training PPE è un limite atteso.
2. **Severità dell'IoU** — effetto *moderato*. mAP@.3 è ~6 punti sopra mAP@.5 (GD 0.576 vs 0.516):
   molti box sono "quasi giusti" ma non centrano l'IoU 0.5, specie gli oggetti piccoli. Per una
   detection di sicurezza grossolana, IoU 0.3 è più rappresentativo.
3. **Soglia di produzione** — effetto *piccolo*. Ottimizzando la sola soglia il macro-F1 di GD sale
   solo da 0.547 a 0.577 (+0.03); l'unica classe che ne beneficia davvero è Shoe. Quindi le soglie
   alte tarate sui video **non** stavano rovinando il recall.

**Conclusione della diagnosi**: il tetto è lo zero-shot sugli oggetti piccoli, non un problema di
configurazione. Questo indirizza i tentativi: prompt/soglie/tiling difficilmente basteranno; serve
o un detector migliore o il training.

---

## 5. Passo 3 — I tentativi di miglioramento

Ogni tentativo è misurato offline sullo stesso subset (`eval_experiments.py` + `eval_compare.py`,
dati in `eval_experiments_results.json`). Riassunto:

| Variante | mAP@.5 | mAP@.3 | macro-F1 (soglia ottimale) |
|---|---|---|---|
| GD baseline | 0.519 | 0.586 | 0.577 |
| GD + prompt multi-sinonimo | 0.521 | 0.580 | 0.598 |
| OmDet baseline | 0.482 | 0.555 | 0.531 |
| OmDet + prompt multi-sinonimo | 0.476 | 0.546 | 0.528 |
| OmDet + SAHI (tiling) | 0.406 | 0.476 | 0.483 |
| OWLv2 (backend alternativo) | 0.583 | 0.626 | 0.622 |
| **🏆 Ensemble OWLv2 + GD (occhiali)** | **0.629** | **0.683** | — † |

*(NMS per-classe attiva su tutte le varianti. GD baseline qui è 0.519 vs 0.516 del §3 per via della
NMS — vedi 5.1. † F1 non riportata per l'ensemble: i due detector hanno scale di score diverse.)*

**AP@.5 per classe** (le più significative):

| Variante | Person | Helmet | Vest | Glasses | Glove | Shoe |
|---|---|---|---|---|---|---|
| GD baseline | 0.81 | 0.59 | 0.48 | 0.50 | 0.40 | 0.34 |
| GD multi-sinonimo | 0.79 | 0.52 | 0.49 | 0.44 | 0.33 | **0.55** |
| OWLv2 | 0.79 | **0.75** | **0.65** | 0.23 | **0.63** | 0.44 |
| **Ensemble** | 0.79 | **0.75** | **0.65** | **0.50** | **0.63** | 0.44 |

### 5.1 NMS per-classe — *perché*: togliere i box duplicati sovrapposti (nei video se ne vedevano)
Aggiunta `nms_per_class` (IoU 0.5) in `analyzer.py`. **Esito: trascurabile sulle metriche**
(GD mAP@.5 0.516→0.519, OmDet 0.482→0.482). Motivo: la valutazione mAP già "assorbe" i duplicati
(diventano FP a bassa confidence) e le over-detection sono per lo più box *spazialmente distinti*,
non duplicati stacked (la NMS rimuove solo 3 box su OmDet, 154 su GD). **Tenuta comunque** perché
pulisce l'output visivo. → *non* è la leva per le metriche.

### 5.2 Prompt multi-sinonimo — *perché*: la letteratura riporta fino a 2× mAP con più frasi per classe
Provate più formulazioni per classe (es. "safety glasses" + "protective eyewear"). **Esito:
marginale e ridistributivo.** Su GD +0.02 F1 (unico guadagno vero: Shoe AP 0.34→0.55, grazie a
"work boot" + "protective footwear"); su OmDet neutro/negativo. → non un miglioramento netto.

### 5.3 SAHI (tiling) — *perché*: tecnica standard per oggetti piccoli (i nostri guanti/scarpe)
Inferenza su tile sovrapposte + immagine intera, poi fusione (`sahi_infer.py`). **Esito: peggiora**
(mAP 0.48→0.41). Su immagini **già piccole (640px)** il tiling spezza gli oggetti grandi: la
detection di Person crolla (AP 0.78→0.40) e domina il calo. SAHI serve su immagini grandi, non qui.

### 5.4 OWLv2 — *perché*: provare un detector open-vocabulary **alternativo** con la stessa licenza
Aggiunto `Owlv2Analyzer` (`google/owlv2`, **Apache 2.0**, nativo in transformers). **Esito: è il
miglior singolo detector** — mAP@.5 **0.583** (+0.064 su GD), nettamente meglio su Helmet (0.75),
Vest (0.65), Glove (0.63); ~5.8 s/img (tra OmDet e GD). **Debole solo sugli occhiali (AP 0.23).**
Ha una scala di score diversa (sigmoide) → soglie per-classe proprie (`OWLV2_CONF`); con quelle è
anche il miglior F1 (0.622).

### 5.5 MM-Grounding-DINO — *perché*: reimplementazione migliorata di Grounding DINO (Apache 2.0)
**Non valutato**: richiede lo stack mmdetection/mmcv, non installabile in modo affidabile su questo
ambiente Windows/CPU. Rimane un candidato per il futuro.

### 5.6 Ensemble OWLv2 + Grounding DINO — *perché*: coprire l'unico punto debole di OWLv2 (occhiali)
`EnsembleAnalyzer`: OWLv2 per tutte le classi + Grounding DINO solo per gli occhiali (dove GD è il
migliore). **Esito: la configurazione migliore in assoluto** — mAP@.5 **0.629** (Glasses AP
0.23→0.50, il resto invariato), **+0.11 mAP (+21%) sul GD baseline**. Costo: somma delle due
inferenze (~15 s/img su CPU) → accuracy-first, non real-time. `build_detector("ensemble")`.

### 5.7 Fine-tuning — *perché*: sarebbe il vero salto (letteratura: +15 mAP con poche centinaia di img)
**Rimandato**: su CPU non praticabile (full fine-tune ~40 h/epoca GD, ~7 h/epoca OmDet). Fattibile
solo su GPU (Colab ~2 h). È la strada indicata per alzare davvero occhiali e oggetti piccoli.

---

## 6. Passo 4 — Verifica sui video demo (OWLv2 e ensemble)

Rigenerati i video su `ppe_video.mp4` con i nuovi backend, per confrontarli con §3 anche sul
comportamento reale (e sui falsi positivi, dato che in quel video occhiali/sigarette **non esistono**):

| Detector | FPS effettivi | Alert | Track | Glasses FP | Sigarette FP |
|---|---|---|---|---|---|
| OmDet-Turbo | 5.05 | 20 | 5 | 0 | 0 |
| Grounding DINO | 0.59 | 13 | 3 | 0 | 0 |
| OWLv2 | 1.31 | 12 | 5 | **4** | 0 |
| **Ensemble** | 0.44 | 14 | 5 | **0** | 0 |

- **L'ensemble è pulito anche sul video** (0 FP): gli occhiali arrivano da GD, che ha la soglia già
  tarata (0.45). **OWLv2 da solo produce 4 FP occhiali** perché le sue soglie sono tarate sull'F1 di
  SH17, non sui FP di un video → se si usasse OWLv2 **da solo** in produzione andrebbero ri-tarate
  (per l'ensemble non serve).
- **Velocità**: OWLv2 (1.31 FPS) sta tra OmDet e GD; l'ensemble (0.44 FPS) è il più lento.
- Video: `output/test_tracker_{owlv2,ensemble}.mp4`.

---

## 7. Conclusioni e raccomandazioni

- **Qual è meglio, GD o OmDet?** GD è più accurato, OmDet molto più veloce. Ma la scoperta è che
  **c'è di meglio di entrambi**: OWLv2 (stessa licenza) e soprattutto l'**ensemble OWLv2+GD**.
- **Per il caso d'uso**:
  - *Sorveglianza live su CPU* → **OmDet-Turbo** (unico ~real-time), con sliding-window + check
    geometrici a valle per contenere i FP.
  - *Analisi offline / massima accuratezza* → **ensemble OWLv2+GD** (mAP@.5 0.629).
  - *Compromesso accuratezza/velocità* → **OWLv2** (ma ri-tarare le soglie sul video).
- **Per un salto ulteriore** (occhiali, oggetti piccoli): **fine-tuning su GPU** — l'unica leva che,
  secondo la letteratura, sposta davvero il tetto dello zero-shot.

---

## 8. Limiti e note metodologiche (onestà sui numeri)

- **AP a IoU singolo, non COCO mAP@[.5:.95]**: riportiamo AP@.5 e AP@.3 (più informativi per detection
  di sicurezza grossolana), non la media COCO su 10 soglie.
- **Check geometrici non applicati nella statica**: i filtri di plausibilità (occhiali in fascia-volto,
  dimensione/larghezza sigaretta) sono logica di *sistema* a valle e non attivi nel test detection-level
  → i FP statici (specie Cigarette) sovrastimano quelli del pipeline completo.
- **Domain gap classi**: "safety glasses"/"work glove"/"work boot" (prompt) vs occhiali/guanti/scarpe
  generici (SH17) penalizzano il recall — limite dello zero-shot, non necessariamente del modello.
- **Recall Person < 1**: SH17 etichetta persone minuscole/di sfondo che la soglia scarta; non grave
  per il caso d'uso (operai in primo piano).
- **Soglie OWLv2 provvisorie**: `OWLV2_CONF` è tarata sull'F1 di SH17, non sui FP di un video reale.
- **Cigarette non valutabile staticamente**: nessun dataset PPE la annota; validazione solo qualitativa
  sui video.

---

## 9. File prodotti e riproducibilità

```
evaluation/
├── REPORT.md                        # questo report
├── build_subset.py                  # subset bilanciato SH17 (seed 42)
├── eval_capture.py                  # cattura detection grezze baseline (soglia bassa)
├── eval_metrics.py                  # metriche offline dal dump: mAP@.5/.3, F1 prod vs best, sweep
├── eval_experiments.py              # cattura varianti Tier 1/2 (multi-sinonimo, SAHI, OWLv2)
├── eval_compare.py                  # confronto mAP di tutte le varianti + ensemble (offline)
├── sahi_infer.py                    # helper SAHI (inferenza a tile)
├── eval_static.py                   # (legacy) P/R/F1 diretto al punto operativo
├── temporal_stats.py                # statistiche temporali sui video
├── eval_metrics_results.json        # baseline (no_nms/nms), per-classe
├── eval_experiments_results.json    # varianti Tier 1/2 + ensemble
├── temporal_stats_results.json      # FPS/latenza/track/alert
├── raw_detections_*.json            # dump detection (baseline versionati; esperimenti gitignored)
├── sh17_subset/{images[gitignored], ground_truth.json}
├── sh17_meta/ [gitignored]          # cache download COCO
└── verify/ [gitignored]             # crop verifica classi + montaggi
```

```bash
python evaluation/build_subset.py      # ricostruisce il subset (serve HF_TOKEN nel .env)
python evaluation/eval_capture.py      # cattura baseline GD/OmDet (~45 min su CPU)
python evaluation/eval_metrics.py      # metriche baseline (istantaneo)
python evaluation/eval_experiments.py  # cattura varianti Tier 1/2 (~1.5 h su CPU)
python evaluation/eval_compare.py      # confronto di tutte le varianti + ensemble (istantaneo)
python evaluation/temporal_stats.py    # statistiche temporali sui video
```

**Attribuzione dataset**: SH17 — https://universe.roboflow.com/safety-measure/sh17-dataset — CC BY 4.0.
