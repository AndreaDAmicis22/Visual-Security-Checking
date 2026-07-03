panoramica di tutti i file .py in src/visual_security/, organizzati per ruolo nella pipeline:

Package root
__init__.py — Espone la API pubblica del package: AnalysisResult, BaseAnalyzer, Detection, YOLOAnalyzer (da analyzer.py), PersonPPEChecker/PersonPPEResult (da person_ppe_checker.py), VideoSafetyTracker/build_tracker (da video_tracker.py), LocalVLMValidator (da vlm_validator.py).

__main__.py — Entry point per python -m visual_security; richiama semplicemente cli.main().

cli.py — Interfaccia a riga di comando basata su argparse, con due sottocomandi:

track — avvia il tracking video real-time (YOLO + PPEChecker + VLM opzionale) tramite build_tracker(), con opzioni per modello YOLO, sorgente video/webcam, modello VLM, soglie di persistenza/finestra, salvataggio output annotato e log alert.
check-vlm — verifica se torch/transformers sono installati e quindi se il backend VLM locale è disponibile.
Utility e dati
utils/paths.py — Definisce i path canonici del progetto (ROOT_DIR, DATA_DIR, YOLO_DIR, SRC_DIR) risalendo dalla posizione del file, più un helper get_data_yaml().

download_data.py — Script standalone che scarica il dataset da Roboflow (progetto akfa-beqxl/safety-rd-v1, formato YOLOv11) usando l'API key da .env, e lo sposta in DATA_DIR.

training.py — Script di training YOLO11 (Ultralytics) con parametri di augmentation estesi (mosaic, mixup, copy-paste, HSV, ecc.), poi esporta il modello in ONNX (con NMS incorporato) al termine.

Core pipeline
analyzer.py — Definisce i modelli dati (Detection, AnalysisResult) e le etichette (VIOLATION_LABELS, PPE_LABELS). BaseAnalyzer è la classe astratta con timing/error-handling standardizzato; YOLOAnalyzer è l'implementazione concreta che carica un backend ONNX Runtime locale (dal sottopackage yolo/) ed esegue l'inferenza restituendo Detection tipizzate.

person_ppe_checker.py — Logica di associazione persona↔DPI: normalizza qualunque formato di bbox (xyxy/cxcywh, normalizzato/pixel, poligono) in _to_xyxy(), calcola overlap con containment+IoU, e per ogni Person rilevata verifica tramite assegnazione greedy quali DPI richiesti (Helmet, Vest, Glove×2, Shoe×2) sono presenti, espandendo la bbox della persona per i DPI che escono dal corpo (guanti, scarpe, casco). Produce PersonPPEResult con stato di conformità.

vlm_validator.py — Secondo stadio di validazione via VLM locale in-process (SmolVLM-500M/2.2B, non Ollama/CLIP/moondream2 — la docstring motiva la scelta, coerente con la mia memoria salvata su questo progetto). Carica lazy il modello via transformers, e per ogni DPI marcato mancante fa una domanda yes/no al VLM sul crop della persona per confermare o scartare il falso positivo.

video_tracker.py — Orchestratore principale: VideoViolationTracker implementa una sliding-window per cella di griglia + tipo di violazione per confermare solo violazioni persistenti (riduce falsi positivi da detection singole mancate); VideoSafetyTracker gestisce il loop video (cattura, YOLO ogni N frame, persistence filtering, validazione VLM asincrona su thread separato per non bloccare il display, disegno overlay/HUD, salvataggio video annotato e log JSON); build_tracker() è la factory che assembla tutti i componenti da parametri semplici (usata da cli.py).

Script di debug
debug_frame.py — Diagnostica su singola immagine: stampa le detection YOLO grezze con bbox, esegue PersonPPEChecker mostrando le associazioni persona↔DPI, e salva un'immagine annotata su disco per tarare le soglie senza eseguire l'intero loop video.

debug_video.py (file aperto nell'IDE) — Campiona N frame equidistanti da un video ed esegue YOLO su ciascuno, stampando le detection raw insieme a diagnostica sul formato bbox (rileva se normalizzato 0-1, o probabile formato cx,cy,w,h) per aiutare a capire cosa restituisce il backend, senza scrivere nulla su disco.