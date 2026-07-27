# Visual Security — PPE Tracker

Real-time PPE (Personal Protective Equipment) tracker per cantieri.

Verifica per ogni persona i **DPI richiesti** (casco, gilet, occhiali, guanti,
scarpe) e la presenza di **item vietati** (sigarette).

Pipeline: **detector open-vocabulary** (Grounding DINO / OmDet-Turbo / OWLv2, o **ensemble**, zero-shot) → **PersonPPEChecker** (associazione DPI↔persona) → **PersonTracker** (identità + memoria PPE) → **sliding window** (conferma violazioni persistenti) → video annotato + log JSON.

> Per l'architettura completa, il ruolo di ogni file, i parametri di taratura e le decisioni di design: **[INFO.md](INFO.md)**.

## Licenze — perché niente YOLO/Ultralytics

Tutti i modelli usati sono **Apache 2.0** e girano in-process via `transformers`:
il codice che li usa **può restare proprietario**. Ultralytics (YOLOv8/11) è
**AGPL-3.0**: metterla in produzione obbligherebbe a pubblicare il codice
sorgente dell'applicazione. Per questo è stata rimossa completamente.

| Componente | Modello | Licenza |
|---|---|---|
| Detection (real-time su CPU) | [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) | Apache 2.0 |
| Detection (miglior mAP singolo) | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Apache 2.0 |
| Detection (max accuratezza zero-shot) | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | Apache 2.0 |
| **Ensemble** (accuratezza top) | OWLv2 + Grounding DINO (occhiali) | Apache 2.0 |

I detector sono **open-vocabulary**: rilevano le classi da prompt testuali
("a person", "a hard hat", "safety glasses", "a cigarette", ...) **senza alcun
training** — niente dataset PPE da trovare/etichettare, niente fine-tuning.
Aggiungere una classe = aggiungere una frase in `DETECTION_PROMPTS` (analyzer.py).
Tutti e quattro i backend sono selezionabili (`--detector`, o dalla web UI).

> Nota storica: la pipeline includeva uno stadio di validazione VLM (SmolVLM)
> nato per compensare una YOLO addestrata male su guanti e scarpe. Con i
> detector open-vocabulary era diventato ridondante ed è stato rimosso
> (dimezzando i tempi di runtime).

## Setup

```bash
# Installa tutte le dipendenze (torch + transformers + timm)
pip install -r requirements.txt
```

Nessun server esterno: tutto gira **in-process**. I pesi dei modelli vengono
scaricati automaticamente da HuggingFace al primo utilizzo (~1GB).

## Usage

### Web UI (browser)

Interfaccia grafica per caricare un video in **drag-and-drop**, scegliere il modello
e i parametri, avviare il tracking e **scaricare il video annotato**.

```bash
pip install -r requirements.txt        # include FastAPI + uvicorn
python webapp/app.py                    # -> http://127.0.0.1:8000
```

Nella pagina: scegli il detector (OmDet-Turbo veloce · OWLv2 miglior mAP · Grounding
DINO max accuratezza · Ensemble OWLv2+GD), regola le impostazioni avanzate
(skip-frames, persistenza, window, memoria PPE, confidence), trascina il video e
avvia. Il processing gira come **job asincrono** con barra di avanzamento; a fine
elaborazione trovi il riepilogo alert e il pulsante di **download** dell'output.

> Su CPU l'elaborazione è lenta (da ~1 min con OmDet a ~9 min con l'ensemble per un
> video di ~10 s) — la barra mostra l'avanzamento reale per frame.

### CLI

```bash
# Track a video
python -m visual_security.cli track --source video.mp4

# Track completo: output annotato + log
python -m visual_security.cli track \
    --source video.mp4 \
    --save-output output/annotated.mp4 \
    --alert-log output/alerts.json

# Scelta del detector: omdet-turbo (veloce) | owlv2 (miglior mAP) |
# grounding-dino (max accuratezza) | ensemble (owlv2 + grounding-dino sugli occhiali)
python -m visual_security.cli track --source 0 --detector omdet-turbo
python -m visual_security.cli track --source video.mp4 --detector ensemble

# Verifica che il backend (torch/transformers) sia disponibile
python -m visual_security.cli check-backend
```

### Cosa viene verificato

| Categoria | Voci | Regola |
|---|---|---|
| DPI richiesti | Casco ×1, Gilet ×1, Occhiali ×1, Guanti ×2, Scarpe ×2 | violazione se **manca** |
| Item vietati | Sigaretta | violazione se **presente** |

Le quantità richieste sono in `REQUIRED_PPE_COUNTS` e gli item vietati in
`PROHIBITED_ITEMS` (entrambi in `person_ppe_checker.py`). In entrambi i casi
l'alert scatta solo se la condizione **persiste** per N frame (sliding window).

### Notebook

- `test_tracker.ipynb` — esegue la pipeline completa su un video di test: sampling
  dei frame con detection raw, tracking completo, analisi visiva degli alert e
  report del log JSON.
- `benchmark_tracker.ipynb` — **confronto quantitativo** Grounding DINO vs
  OmDet-Turbo: tracker end-to-end sul video (FPS, stabilità identità, alert),
  metriche frame-level contro weak ground truth dichiarata, e Precision/Recall/F1
  su **`benchmark_data/`** — 500 immagini etichettate incluse nel progetto
  (subset stratificato del dataset Roboflow "PPE Combined Model", CC BY 4.0,
  valutazione class-aware per via delle annotazioni parziali — vedi
  `benchmark_data/README.md`).

### Script di debug

```bash
# Diagnostica su singola immagine (detection + associazioni DPI, salva immagine annotata)
python -m src.visual_security.debug_frame --image frame.jpg --detector grounding-dino

# Diagnostica detection raw su N frame campionati dal video
python -m src.visual_security.debug_video --video video.mp4 --detector omdet-turbo --samples 6
```

## Valutazione e performance

Confronto quantitativo completo (accuratezza su dataset etichettato + velocità sui
video) in **[evaluation/REPORT.md](evaluation/REPORT.md)** — racconto lineare di tutte
le prove fatte. Subset bilanciato di **259 immagini SH17**, CPU-only (Intel Iris Xe, 16 GB).

| Detector | mAP@.5 | mAP@.3 | FPS eff. (CPU) | Quando usarlo |
|---|---|---|---|---|
| OmDet-Turbo | 0.482 | 0.555 | **5.05** | sorveglianza **live** su CPU |
| Grounding DINO | 0.519 | 0.586 | 0.59 | analisi offline / GPU |
| OWLv2 | 0.583 | 0.626 | 1.31 | miglior **detector singolo** |
| **Ensemble OWLv2+GD** | **0.629** | **0.683** | 0.44 | **massima accuratezza** (offline) |

- **Cuore del sistema solido**: Person AP@.5 ~0.80. Il calo è sugli oggetti PPE piccoli
  (guanti/scarpe/occhiali) — limite atteso dello **zero-shot senza training**.
- **OWLv2** (Apache 2.0) batte GD/OmDet su casco/gilet/guanti; l'**ensemble** (OWLv2 +
  Grounding DINO sugli occhiali) è il migliore in assoluto: **+21% mAP sul baseline GD**,
  e 0 falsi positivi sui video.
- Prompt multi-sinonimo e SAHI (tiling) provati ma **non utili**; il prossimo salto reale
  sarebbe il **fine-tuning** (non praticabile su CPU, serve GPU).
- Su **GPU** i detector singoli scendono sotto i ~200 ms/frame.

> Riproducibilità e dettaglio per-classe: cartella `evaluation/` (script + `REPORT.md`).
