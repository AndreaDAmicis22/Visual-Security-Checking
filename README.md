# Visual Security — PPE Tracker

Real-time PPE (Personal Protective Equipment) tracker + monitoraggio aree vietate per cantieri.

Pipeline: **detector open-vocabulary** (Grounding DINO / OmDet-Turbo, zero-shot) → **PersonPPEChecker** (associazione DPI↔persona) → **PersonTracker** (identità + memoria PPE) → **sliding window** (conferma violazioni persistenti) → **ZoneMonitor** (aree vietate) → video annotato + log JSON.

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
("a person", "a hard hat", "a reflective safety vest", ...) **senza alcun
training** — niente dataset PPE da trovare/etichettare, niente fine-tuning.

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

# Track completo: zone vietate + output annotato + log
python -m visual_security.cli track \
    --source video.mp4 \
    --zones zones.example.json \
    --save-output output/annotated.mp4 \
    --alert-log output/alerts.json

# Detector veloce (OmDet-Turbo, ~1.5s/frame su CPU vs ~22s di Grounding DINO)
python -m visual_security.cli track --source 0 --detector omdet-turbo

# Verifica che il backend (torch/transformers) sia disponibile
python -m visual_security.cli check-backend
```

### Aree vietate

Definisci i poligoni in un file JSON (coordinate normalizzate 0-1 o pixel):

```json
{
  "zones": [
    {
      "name": "Area gru",
      "polygon": [[0.05, 0.55], [0.40, 0.55], [0.40, 0.98], [0.05, 0.98]],
      "normalized": true
    }
  ]
}
```

Una persona è "in zona" se il suo **punto-piedi** (centro del lato inferiore
della bbox) cade nel poligono; l'alert scatta solo se la presenza **persiste**
per N frame (stessa sliding window dei PPE). Vedi `zones.example.json`.

### Notebook

- `test_tracker.ipynb` — esegue la pipeline completa su un video di test: sampling
  dei frame con detection raw, tracking completo, analisi visiva degli alert e
  report del log JSON.
- `benchmark_tracker.ipynb` — **confronto quantitativo** Grounding DINO vs
  OmDet-Turbo: tracker end-to-end sul video (FPS, stabilità identità, alert),
  metriche frame-level contro weak ground truth dichiarata, e Precision/Recall/F1
  su un test split etichettato (Roboflow, valutazione stratificata class-aware).

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
