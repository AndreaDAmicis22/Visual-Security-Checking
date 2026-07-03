# Visual Security — PPE Tracker

Real-time PPE (Personal Protective Equipment) tracker for construction sites.

Pipeline: **YOLO** (detection ONNX) → **PersonPPEChecker** (associazione DPI↔persona) → **PersonTracker** (identità + memoria PPE) → **sliding window** (conferma violazioni persistenti) → **VLM locale** (validazione crop, escalation) → video annotato + log JSON.

> Per l'architettura completa, il ruolo di ogni file, i parametri di taratura e le decisioni di design: **[INFO.md](INFO.md)**.

## Setup

```bash
# Installa tutte le dipendenze (incluso il backend VLM: torch + transformers)
pip install -r requirements.txt
```

Nessun server esterno: il VLM gira **in-process**. I pesi del modello
([SmolVLM-500M](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct), ~1GB)
vengono scaricati automaticamente da HuggingFace al primo utilizzo.

## Usage

### CLI

```bash
# Track a video (YOLO only, no VLM)
python -m visual_security.cli track \
    --yolo-model weights/dataset_1/yolo_nano_640/best.onnx \
    --source video.mp4 \
    --no-vlm

# Track con escalation VLM locale (default: SmolVLM-500M)
python -m visual_security.cli track \
    --yolo-model weights/best.onnx \
    --source video.mp4 \
    --vlm-model HuggingFaceTB/SmolVLM-500M-Instruct \
    --save-output output/annotated.mp4 \
    --alert-log output/alerts.json

# Webcam
python -m visual_security.cli track \
    --yolo-model weights/best.onnx \
    --source 0

# Verifica che il backend VLM (torch/transformers) sia disponibile
python -m visual_security.cli check-vlm
```

Per più accuratezza (a costo di più tempo su CPU e ~9GB di pesi):
`--vlm-model HuggingFaceTB/SmolVLM2-2.2B-Instruct`.

### Notebook

`test_tracker.ipynb` esegue la pipeline completa su un video di test: sampling
dei frame con detection raw, tracking completo con VLM, analisi visiva degli
alert e report del log JSON.

### Script di debug

```bash
# Diagnostica su singola immagine (detection + associazioni DPI, salva immagine annotata)
python -m src.visual_security.debug_frame --image frame.jpg --yolo-model weights/best.onnx

# Diagnostica detection raw su N frame campionati dal video
python -m src.visual_security.debug_video --video video.mp4 --yolo-model weights/best.onnx --samples 6
```

### Training

```bash
# Scarica il dataset da Roboflow (richiede ROBOFLOW_API_KEY in .env)
python -m src.visual_security.download_data

# Addestra YOLO11 ed esporta in ONNX
python -m src.visual_security.training
```
