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
                                    │  Ollama VLM (locale)   │  ← escalation
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
Il **VLM (Ollama)** valida il crop della persona con domande yes/no — viene chiamato solo sulle violazioni confermate dal tracker (escalation, non ogni frame).

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama (for VLM escalation)
# Download from https://ollama.com/download
# Then pull a lightweight vision model:
ollama pull moondream      # 1.6B, fastest
# or
ollama pull minicpm-v      # 3B, more accurate
# or
ollama pull llava-phi3     # 3.8B, best accuracy
```

## Usage

```bash
# Track a video (YOLO only)
python -m visual_security.cli track \
    --yolo-model weights/dataset_1/yolo_nano_640/best.onnx \
    --source video.mp4

# Track with VLM escalation
python -m visual_security.cli track \
    --yolo-model weights/best.onnx \
    --source video.mp4 \
    --vlm moondream \
    --save-output output/annotated.mp4 \
    --alert-log output/alerts.json

# Webcam
python -m visual_security.cli track \
    --yolo-model weights/best.onnx \
    --source 0

# Verify Ollama setup
python -m visual_security.cli check-vlm --model moondream
```

## Why Ollama?

- **Gratuito**: nessuna API key, nessun costo per chiamata
- **Locale**: i dati non lasciano mai la macchina
- **Leggero**: moondream gira su CPU in ~2-3s per query (GPU: <500ms)
- **Semplice**: un binario, `ollama pull`, pronto

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