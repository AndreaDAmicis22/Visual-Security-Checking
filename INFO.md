# INFO — Visual Security PPE Tracker

Guida di riferimento interna: architettura della pipeline, ruolo di ogni file e parametri di taratura.

---

## 1 · Architettura

```
┌──────────────────────┐  ogni N frame   ┌────────────────────────┐
│ Detector open-vocab  │ ─ detections ──►│  PersonPPEChecker      │
│ (Grounding DINO /    │                 │  (containment overlap) │
│  OmDet-Turbo)        │                 └───────────┬────────────┘
└──────────────────────┘                             │ PersonPPEResult list
                                                      │ (mancanti + vietati)
                                         ┌───────────▼────────────┐
                                         │  PersonTracker         │
                                         │  (identità IoU +       │
                                         │   memoria PPE)         │
                                         └───────────┬────────────┘
                                                     │ track_id + missing smussati
                                         ┌───────────▼────────────┐
                                         │  VideoViolationTracker │
                                         │  (sliding window)      │
                                         └───────────┬────────────┘
                                                     │ violazioni confermate
                                         ┌───────────▼────────────┐
                                         │  Output annotato       │
                                         │  (video / JSON log)    │
                                         └────────────────────────┘
```

Principio chiave: **il detector lavora a oggetti singoli** (ogni guanto/scarpa è una detection indipendente), **il checker aggrega per categoria** (Glove 2/2, Shoe 2/2) e distingue **DPI richiesti** (violazione se mancano) da **item vietati** (violazione se presenti, es. sigarette).

> Nota storica: fino a luglio 2026 esisteva uno stadio di validazione VLM (SmolVLM) come seconda
> opinione sugli alert — serviva a compensare una YOLO addestrata male che non rilevava quasi mai
> guanti e scarpe. Con i detector open-vocabulary è ridondante ed è stato rimosso.

### Vincolo di licenza (il motivo di questa architettura)

Nessun componente Ultralytics/YOLO: la licenza **AGPL-3.0** di Ultralytics obbligherebbe a pubblicare il codice dell'applicazione in produzione. Tutti i modelli sono **Apache 2.0** e girano **nativamente in `transformers`** (no `trust_remote_code`):

| Ruolo | Modello | Note |
|---|---|---|
| Detection (default) | `IDEA-Research/grounding-dino-base` | max accuratezza zero-shot, ~22s/frame su CPU |
| Detection (fast) | `omlab/omdet-turbo-swin-tiny-hf` | ~1.5s/frame su CPU, richiede `timm` |

I detector sono **open-vocabulary**: le classi arrivano da prompt testuali (`DETECTION_PROMPTS` in `analyzer.py`) — zero training, zero dataset. Questo aggira il collo di bottiglia storico del progetto (nessun dataset PPE adeguato).

---

## 2 · Pipeline, stadio per stadio

### 2.1 Loop video — `VideoSafetyTracker.run()`
Legge la sorgente frame per frame (file o webcam). Il detector gira solo ogni `skip_frames` frame; nei frame intermedi si riusano gli ultimi risultati. Ogni frame viene comunque annotato e scritto nel video di output.

### 2.2 Detection — `GroundingDinoAnalyzer` / `OmDetTurboAnalyzer`
Inferenza zero-shot in-process: rileva oggetti **singoli** — Person, Helmet, Vest, Glasses, ogni singolo Glove, ogni singola Shoe, più gli item vietati (Cigarette) — da prompt testuali, con bbox pixel e confidence. Il testo matchato dal modello viene riportato alla categoria canonica tramite keyword (`_match_label`).

### 2.3 Associazione — `PersonPPEChecker`
Per ogni frame, assegna ogni oggetto alla persona con l'overlap maggiore (containment + IoU, con bbox della persona espansa per guanti/scarpe/casco/sigarette che sporgono dal corpo). Conta i DPI per categoria e li confronta con i requisiti (`REQUIRED_PPE_COUNTS`: Helmet 1, Vest 1, Glasses 1, Glove 2, Shoe 2) → lista `missing_ppe`; separatamente registra gli item vietati (`PROHIBITED_ITEMS`: Cigarette) trovati addosso → lista `prohibited_present`. La proprietà `violation_labels` unisce le due liste. È **stateless**.

### 2.4 Identità + memoria — `PersonTracker`
Associa le persone tra frame consecutivi per IoU e assegna un `track_id` stabile (mostrato come `[T3]`). Per ogni persona ricorda i DPI visti di recente: un guanto osservato negli ultimi `ppe_memory_frames` frame è considerato ancora presente anche se il detector lo perde per occlusione/blur. È il filtro anti-violazioni-fantasma.

### 2.5 Persistenza — `VideoViolationTracker`
Sliding window per chiave `track_id + violation_labels` (DPI mancanti **e** item vietati presenti): la violazione viene **confermata** solo se appare in almeno `persistence_frames` degli ultimi `window_frames`, poi entra in cooldown (30 frame). La cella di griglia resta solo come fallback per persone senza track.

### 2.6 Output
- **Video annotato**: verde = OK, gradiente giallo→rosso = violazione in accumulo (con % di riempimento window), rosso = alert confermato; nel riquadro di ogni persona la checklist dei DPI (✔/✘) e le righe rosse `VIETATO <item>` per gli item proibiti presenti; HUD con FPS/contatori.
- **JSON log**: per ogni alert, timestamp, frame, `track_id`, DPI trovati/mancanti, `prohibited_present`, bbox.

---

## 3 · File del package `src/visual_security/`

### Core pipeline
| File | Ruolo |
|---|---|
| `analyzer.py` | Modelli dati (`Detection`, `AnalysisResult`), prompt open-vocabulary (`DETECTION_PROMPTS`), `BaseAnalyzer` (astratta, timing/error-handling), `GroundingDinoAnalyzer` + `OmDetTurboAnalyzer` (transformers, Apache 2.0), factory `build_detector()`. |
| `person_ppe_checker.py` | Associazione persona↔oggetti: normalizzazione bbox universale (`_to_xyxy`), overlap containment+IoU, assegnazione greedy, bbox espansa per oggetti periferici. Distingue DPI richiesti (`REQUIRED_PPE_COUNTS`) da item vietati (`PROHIBITED_ITEMS`). Produce `PersonPPEResult`. |
| `person_tracker.py` | Identità persistente (matching greedy IoU tra frame → `track_id`) + memoria PPE temporale (`ppe_memory_frames`). Riscrive `missing_ppe` con l'evidenza recente. |
| `video_tracker.py` | Orchestratore: `VideoViolationTracker` (sliding window + cooldown per identità), `VideoSafetyTracker` (loop video, disegno, log), `build_tracker()` (factory). |

### Interfacce
| File | Ruolo |
|---|---|
| `__init__.py` | API pubblica del package (analyzer, checker, tracker, factory). |
| `__main__.py` | Entry point `python -m visual_security` → `cli.main()`. |
| `cli.py` | CLI `argparse` con sottocomandi `track` (tracking real-time: `--detector`, soglie) e `check-backend` (verifica torch/transformers). |

### Utility e dati
| File | Ruolo |
|---|---|
| `utils/paths.py` | Path canonici del progetto (`ROOT_DIR`, `DATA_DIR`, ...). |
| `download_data.py` | Download del dataset Roboflow completo (API key da `.env`) — serve solo per **ricostruire/estendere** `benchmark_data/`; il benchmark usa il set già incluso. |
| `benchmark_data/` (root) | **Set di valutazione incluso nel progetto**: 500 immagini etichettate (YOLO format), stratificate per classe (Person 150 / Glove 150 / Vest 120 / Helmet 80), CC BY 4.0. Vedi il suo README. |

### Debug
| File | Ruolo |
|---|---|
| `debug_frame.py` | Diagnostica su singola immagine: detection raw, associazioni persona↔DPI, immagine annotata su disco. |
| `debug_video.py` | Campiona N frame dal video, stampa detection raw + diagnostica formato bbox. Non scrive su disco. |

### Notebook
| File | Ruolo |
|---|---|
| `test_tracker.ipynb` | Test qualitativo della pipeline su un video. |
| `benchmark_tracker.ipynb` | Confronto quantitativo tra i detector: tracker end-to-end sul video, frame-level vs weak GT, P/R/F1 su `benchmark_data/` (stratificato class-aware). |

---

## 4 · Parametri di taratura

| Parametro | Default | Effetto |
|---|---|---|
| `detector` | grounding-dino | `omdet-turbo` per 15× più velocità su CPU (leggermente meno accurato). |
| `detector_conf` | backend default (0.35 GD / 0.30 OmDet) | Soglia confidence. Più bassa = più detection (e più rumore). |
| `skip_frames` | 1 | Detector ogni N frame. Su CPU con grounding-dino usare 8+. |
| `ppe_memory_frames` | 48 (~2s) | Memoria PPE del PersonTracker. Più alto = meno falsi allarmi ma sistema più "indulgente". |
| `persistence_frames` / `window_frames` | 4 / 7 | Una violazione è confermata se presente in ≥ persistence degli ultimi window frame. |

**Nota sensibilità**: memoria PPE e persistenza si sommano — con `skip_frames` alto (necessario per grounding-dino su CPU) abbassare `persistence_frames` (es. 3/6), altrimenti su clip brevi non si conferma nulla.

---

## 5 · Perché queste scelte di modello?

### Detector open-vocabulary (e non un detector COCO o YOLO)
- **Licenza**: Grounding DINO e OmDet-Turbo sono Apache 2.0 → uso commerciale senza obblighi di pubblicazione del codice (Ultralytics è AGPL).
- **Zero-shot**: rilevano "hard hat", "safety vest", "work glove" da prompt testuali senza training → aggira il problema storico del dataset PPE.
- **Nativi in transformers**: no `trust_remote_code`, compatibili con le versioni correnti della libreria.
- I detector COCO-pretrained (RT-DETR, D-FINE — pure Apache 2.0) conoscono solo "person": avrebbero richiesto fine-tuning su un dataset PPE.

### Perché il VLM di validazione è stato rimosso
Lo stadio SmolVLM era una **seconda opinione** sulle violazioni confermate: esisteva perché la
vecchia YOLO (mal addestrata per mancanza di dataset) non rilevava quasi mai guanti e scarpe e
produceva falsi positivi sistematici. Con i detector open-vocabulary la qualità della detection non
giustifica più il costo (~2.5s/query + ~1GB di pesi + thread dedicato): rimosso a luglio 2026.

---

## 6 · Note operative

- **Console Windows (cp1252)**: tutto l'output testuale dei moduli usa solo ASCII (`->`, `>=`) — i glifi unicode nei print causano `UnicodeEncodeError` da terminale (il notebook, UTF-8, non è affetto).
- **Primo run**: i pesi vengono scaricati da HuggingFace al primo caricamento (~1GB grounding-dino-base, ~800MB omdet-turbo).
- **omdet-turbo richiede `timm`** (`pip install timm`, già in requirements).
- **Prompt detection**: per aggiungere/modificare le classi rilevate, editare `DETECTION_PROMPTS` in `analyzer.py` (frasi brevi e concrete funzionano meglio: "a hard hat" > "helmet").
- **Set di valutazione**: `benchmark_data/` (500 immagini, incluso nel repo con eccezione nel `.gitignore`) deriva dal dataset Roboflow "PPE Combined Model", che ha **annotazioni parziali** (merge di più dataset — molte immagini annotano una sola classe). Il benchmark usa una valutazione stratificata class-aware; non usarlo per training/valutazioni naive. Nessuna immagine con scarpe annotate → `Shoe` non valutabile staticamente.
