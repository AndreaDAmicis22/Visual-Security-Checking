# Visual Security — PPE Tracker

Real-time PPE (Personal Protective Equipment) tracker for construction sites.

## Architecture

```
┌────────────────┐  ogni N frame    ┌────────────────────────┐
│  YOLO (veloce) │ ─ detections ──► │  PersonPPEChecker      │
└────────────────┘                  │  (containment overlap) │
                                    └───────────┬────────────┘
                                                │ PersonPPEResult list
                                    ┌───────────▼────────────┐
                                    │  VideoViolationTracker │
                                    │  (sliding window)      │
                                    └───────────┬────────────┘
                                                │ violazioni persistenti
                                    ┌───────────▼────────────┐
                                    │  VLM locale (in-proc)  │  ← escalation
                                    │  crop persona + VQA    │
                                    └───────────┬────────────┘
                                                │ FrameAlert
                                    ┌───────────▼────────────┐
                                    │  Output annotato       │
                                    └────────────────────────┘
```

**YOLO** rileva persone e DPI (Helmet, Vest, Glove, Shoe) in tempo reale.
Il **PPEChecker** associa spazialmente i DPI a ogni persona tramite overlap.
Il **VideoViolationTracker** conferma le violazioni solo se persistono per N frame su M (sliding window), eliminando i falsi positivi transitori.
Il **VLM locale** ([moondream2](https://huggingface.co/vikhyatk/moondream2), ~2B, via `transformers`) valida il crop della persona con domande yes/no — viene chiamato solo sulle violazioni confermate dal tracker (escalation, non ogni frame) e su un **thread in background**, quindi non blocca l'elaborazione dei frame.

## Setup

```bash
# Installa tutte le dipendenze (incluso il backend VLM: torch + transformers)
pip install -r requirements.txt
```

Nessun server esterno: il VLM gira **in-process**. I pesi di moondream2 (~4GB)
vengono scaricati automaticamente da HuggingFace al primo utilizzo.

## Usage

```bash
# Track a video (YOLO only, no VLM)
python -m visual_security.cli track \
    --yolo-model weights/dataset_1/yolo_nano_640/best.onnx \
    --source video.mp4 \
    --no-vlm

# Track con escalation VLM locale (default: moondream2)
python -m visual_security.cli track \
    --yolo-model weights/best.onnx \
    --source video.mp4 \
    --vlm-model vikhyatk/moondream2 \
    --save-output output/annotated.mp4 \
    --alert-log output/alerts.json

# Webcam
python -m visual_security.cli track \
    --yolo-model weights/best.onnx \
    --source 0

# Verifica che il backend VLM (torch/transformers) sia disponibile
python -m visual_security.cli check-vlm
```

## Perché un VLM locale (moondream2)?

- **Generativo**: ragiona sull'immagine e gestisce bene le domande con negazione
  ("è *senza* casco?"), dove i modelli contrastivi (CLIP/DINOv2) sbagliano.
- **Zero-shot**: nessun dataset etichettato richiesto.
- **In-process**: niente server esterno, niente Ollama, niente HTTP/JSON fragile.
- **Locale e gratuito**: i dati non lasciano la macchina, nessuna API key.

> Su CPU una query richiede alcuni secondi: per questo il VLM parte **solo** sulle
> violazioni già confermate dal tracker (con cooldown) ed è eseguito fuori dal loop
> dei frame. Se serve più velocità su CPU, si può passare a un modello più piccolo
> senza modifiche al codice: `--vlm-model HuggingFaceTB/SmolVLM-500M-Instruct`.

## Project Structure

```
src/visual_security/
├── analyzer.py          # YOLO ONNX inference
├── person_ppe_checker.py # Spatial PPE↔Person association
├── video_tracker.py     # Video pipeline + sliding window tracker
├── vlm_validator.py     # Ollama VLM escalation
├── cli.py               # CLI entry point
├── debug_frame.py       # Single-frame diagnostics
├── debug_video.py       # Multi-frame video diagnostics
├── training.py          # YOLO training script (ultralytics)
├── download_data.py     # Dataset download (Roboflow)
└── utils/paths.py       # Path constants
```