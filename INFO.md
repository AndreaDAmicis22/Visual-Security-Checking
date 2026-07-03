# INFO — Visual Security PPE Tracker

Guida di riferimento interna: architettura della pipeline, ruolo di ogni file e parametri di taratura.

---

## 1 · Architettura

```
┌────────────────┐  ogni N frame    ┌────────────────────────┐
│  YOLO (ONNX)   │ ─ detections ──► │  PersonPPEChecker      │
└────────────────┘                  │  (containment overlap) │
                                    └───────────┬────────────┘
                                                │ PersonPPEResult list
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
                                    │  VLM locale (in-proc)  │  ← escalation
                                    │  crop persona + VQA    │
                                    └───────────┬────────────┘
                                                │ FrameAlert
                                    ┌───────────▼────────────┐
                                    │  Output annotato       │
                                    │  (video / JSON log)    │
                                    └────────────────────────┘
```

Principio chiave: **YOLO lavora a oggetti singoli** (ogni guanto/scarpa è una detection indipendente), **il checker aggrega per categoria** (Glove 2/2, Shoe 2/2), **il VLM giudica a livello globale per categoria** ("wearing safety gloves on BOTH hands?").

---

## 2 · Pipeline, stadio per stadio

### 2.1 Loop video — `VideoSafetyTracker.run()`
Legge la sorgente frame per frame (file o webcam). YOLO gira solo ogni `skip_frames` frame; nei frame intermedi si riusano gli ultimi risultati. Ogni frame viene comunque annotato e scritto nel video di output.

### 2.2 Detection — `YOLOAnalyzer`
Inferenza ONNX locale: rileva oggetti **singoli** — Person, Helmet, Vest, ogni singolo Glove, ogni singola Shoe — con bbox e confidence (soglia `yolo_conf`).

### 2.3 Associazione — `PersonPPEChecker`
Per ogni frame, assegna ogni DPI alla persona con l'overlap maggiore (containment + IoU, con bbox della persona espansa per guanti/scarpe/casco che sporgono dal corpo). Conta i DPI per categoria e li confronta con i requisiti (Helmet 1, Vest 1, Glove 2, Shoe 2) → lista `missing_ppe` per persona. È **stateless**: da solo non ricorda nulla tra un frame e l'altro.

### 2.4 Identità + memoria — `PersonTracker`
Associa le persone tra frame consecutivi per IoU e assegna un `track_id` stabile (mostrato come `[T3]`). Per ogni persona ricorda i DPI visti di recente: un guanto osservato negli ultimi `ppe_memory_frames` frame (default 48 ≈ 2s a 24fps) è considerato ancora presente anche se YOLO lo perde per occlusione/blur. Riscrive `missing_ppe` con questi conteggi "effettivi" — è il filtro anti-violazioni-fantasma che compensa la debolezza di YOLO su Glove/Shoe.

### 2.5 Persistenza — `VideoViolationTracker`
Sliding window per chiave `track_id + set di DPI mancanti`: la violazione viene **confermata** solo se appare in almeno `persistence_frames` degli ultimi `window_frames`, poi entra in cooldown (30 frame) per non ri-scattare subito. La cella di griglia resta solo come fallback per persone senza bbox.

### 2.6 Escalation — `LocalVLMValidator`
**Solo quando una violazione viene confermata**, il crop della persona (con 30% di padding) va a SmolVLM su un thread in background: una domanda yes/no per ogni categoria mancante, formulata a livello di coppia dove serve ("wearing safety gloves on BOTH hands?"). Il verdetto aggiorna l'alert (per-persona), il log JSON e la cache dei colori, così i frame successivi disegnano il box già corretto.

### 2.7 Output
- **Video annotato**: verde = OK, gradiente giallo→rosso = violazione in accumulo (con % di riempimento window), rosso = alert confermato, viola = scagionato dal VLM; HUD con FPS/contatori.
- **JSON log**: per ogni alert, timestamp, frame, `vlm_confirmed` aggregato e per-violazione, `track_id`, DPI trovati/mancanti, bbox.

---

## 3 · File del package `src/visual_security/`

### Core pipeline
| File | Ruolo |
|---|---|
| `analyzer.py` | Modelli dati (`Detection`, `AnalysisResult`) ed etichette. `BaseAnalyzer` (astratta, timing/error-handling) + `YOLOAnalyzer` (backend ONNX Runtime dal sottopackage `yolo/`). |
| `person_ppe_checker.py` | Associazione persona↔DPI: normalizzazione bbox universale (`_to_xyxy`), overlap containment+IoU, assegnazione greedy, bbox espansa per DPI periferici. Produce `PersonPPEResult`. |
| `person_tracker.py` | Identità persistente (matching greedy IoU tra frame → `track_id`) + memoria PPE temporale (`ppe_memory_frames`). Riscrive `missing_ppe` con l'evidenza recente. |
| `video_tracker.py` | Orchestratore: `VideoViolationTracker` (sliding window + cooldown per identità), `VideoSafetyTracker` (loop video, VLM asincrono su thread, disegno, log), `build_tracker()` (factory, carica il VLM in modo eager). |
| `vlm_validator.py` | Escalation VLM locale in-process (SmolVLM via `transformers`, no server). Una domanda yes/no per categoria mancante; per Glove/Shoe la domanda è esplicitamente su ENTRAMBI ("on BOTH hands/feet"). Ottimizzazione CPU: `do_image_splitting=False`. |

### Interfacce
| File | Ruolo |
|---|---|
| `__init__.py` | API pubblica del package (analyzer, checker, tracker, VLM, `build_tracker`). |
| `__main__.py` | Entry point `python -m visual_security` → `cli.main()`. |
| `cli.py` | CLI `argparse` con sottocomandi `track` (tracking real-time con tutte le opzioni) e `check-vlm` (verifica disponibilità torch/transformers). |

### Utility e dati
| File | Ruolo |
|---|---|
| `utils/paths.py` | Path canonici del progetto (`ROOT_DIR`, `DATA_DIR`, ...). |
| `download_data.py` | Download dataset da Roboflow (API key da `.env`) e spostamento in `data/`. |
| `training.py` | Training YOLO11 (Ultralytics) con augmentation estese, export finale ONNX con NMS. |

### Debug
| File | Ruolo |
|---|---|
| `debug_frame.py` | Diagnostica su singola immagine: detection raw, associazioni persona↔DPI, immagine annotata su disco. Utile per tarare le soglie senza il loop video. |
| `debug_video.py` | Campiona N frame dal video, stampa detection raw + diagnostica formato bbox (normalizzato? cx,cy,w,h?). Non scrive su disco. |

---

## 4 · Parametri di taratura

| Parametro | Default | Effetto |
|---|---|---|
| `yolo_conf` | 0.30 | Soglia confidence YOLO. Più bassa = più detection (e più rumore). |
| `skip_frames` | 1 | YOLO ogni N frame. Più alto = più veloce, meno reattivo. |
| `ppe_memory_frames` | 48 (~2s) | Memoria PPE del PersonTracker. Più alto = meno falsi allarmi ma sistema più "indulgente"; più basso = più sensibile. |
| `persistence_frames` / `window_frames` | 4 / 7 | Una violazione è confermata se presente in ≥ persistence degli ultimi window frame. |
| `vlm_model` | SmolVLM-500M-Instruct | `"none"` per disabilitare. Alternativa più accurata: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (~9GB, più lento su CPU). |

**Nota sensibilità**: memoria PPE e persistenza si sommano — con `ppe_memory_frames=48` + `persistence=5/10` + `skip=3` il sistema è molto conservativo: su clip brevi può non confermare nessuna violazione (e il VLM, che parte **solo** sulle violazioni confermate, non viene mai chiamato). Per aumentare la sensibilità abbassare prima `ppe_memory_frames` (es. 24), poi `persistence_frames`.

---

## 5 · Perché un VLM locale (SmolVLM)?

- **Generativo**: ragiona sull'immagine e gestisce bene le domande con negazione
  ("è *senza* casco?"), dove i modelli contrastivi (CLIP/DINOv2) sbagliano.
- **Zero-shot**: nessun dataset etichettato richiesto.
- **Nativo in `transformers`**: niente `trust_remote_code`, quindi resta compatibile
  con transformers 5.x (moondream2 invece si rompe con la 5.x).
- **In-process**: niente server esterno, niente Ollama, niente HTTP/JSON fragile.
- **Leggero su CPU**: ~2.5s/query con `do_image_splitting=False` (~99 token immagine
  invece di ~900). La validazione parte **solo** sulle violazioni già confermate dal
  tracker ed è eseguita fuori dal loop dei frame.

---

## 6 · Note operative

- **VLM solo su conferma**: se il log mostra `0 alert confermati`, è normale non vedere nessuna chiamata VLM — il caricamento pesi all'avvio (`[VLM] pronto in Ns`) è solo warm-up.
- **Console Windows (cp1252)**: tutto l'output testuale dei moduli usa solo ASCII (`->`, `>=`) — i glifi unicode nei print causano `UnicodeEncodeError` da terminale (il notebook, UTF-8, non è affetto).
- **Naming modelli**: il modello grande è `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (**SmolVLM2**, con il 2) — `SmolVLM-2.2B-Instruct` non esiste su HuggingFace Hub.
- **Primo run**: i pesi VLM vengono scaricati da HuggingFace al primo caricamento (~1GB per il 500M, ~9GB per il 2.2B).
