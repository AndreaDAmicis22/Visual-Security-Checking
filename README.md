# Visual Security — PPE Tracker

Real-time PPE (Personal Protective Equipment) tracker per cantieri.

Verifica per ogni persona i **DPI richiesti** (casco, gilet, occhiali, guanti,
scarpe) e la presenza di **item vietati** (sigarette).

Pipeline: **detector open-vocabulary** (Grounding DINO / OmDet-Turbo, zero-shot) → **PersonPPEChecker** (associazione DPI↔persona) → **PersonTracker** (identità + memoria PPE) → **sliding window** (conferma violazioni persistenti) → video annotato + log JSON.

> Per l'architettura completa, il ruolo di ogni file, i parametri di taratura e le decisioni di design: **[INFO.md](INFO.md)**.

## Licenze — perché niente YOLO/Ultralytics

Tutti i modelli usati sono **Apache 2.0** e girano in-process via `transformers`:
il codice che li usa **può restare proprietario**. Ultralytics (YOLOv8/11) è
**AGPL-3.0**: metterla in produzione obbligherebbe a pubblicare il codice
sorgente dell'applicazione. Per questo è stata rimossa completamente.

| Componente | Modello | Licenza |
|---|---|---|
| Detection (default, max accuratezza) | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | Apache 2.0 |
| Detection (alternativa real-time) | [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) | Apache 2.0 |

I detector sono **open-vocabulary**: rilevano le classi da prompt testuali
("a person", "a hard hat", "safety glasses", "a cigarette", ...) **senza alcun
training** — niente dataset PPE da trovare/etichettare, niente fine-tuning.
Aggiungere una classe = aggiungere una frase in `DETECTION_PROMPTS` (analyzer.py).

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

### CLI

```bash
# Track a video
python -m visual_security.cli track --source video.mp4

# Track completo: output annotato + log
python -m visual_security.cli track \
    --source video.mp4 \
    --save-output output/annotated.mp4 \
    --alert-log output/alerts.json

# Detector veloce (OmDet-Turbo, ~1.5s/frame su CPU vs ~22s di Grounding DINO)
python -m visual_security.cli track --source 0 --detector omdet-turbo

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

## Performance (CPU-only, Intel Iris Xe, 16GB)

| Backend | Latenza/frame | Note |
|---|---|---|
| grounding-dino (default) | ~22s | massima accuratezza; usare `--skip-frames` alto o GPU |
| omdet-turbo | ~1.5s | qualità comparabile su scene semplici, 15× più veloce |

Su GPU entrambi scendono sotto i 200ms/frame. Per numeri aggiornati sul tuo
hardware esegui `benchmark_tracker.ipynb`.
