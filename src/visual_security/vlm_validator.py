"""
VLM Validator — Escalation via a local, in-process vision LLM.

Quando il tracker YOLO+PPEChecker conferma una violazione persistente,
il crop della persona viene inviato a un piccolo VLM *locale* per una
seconda opinione (riduce i falsi positivi di una YOLO ancora non
addestrata in modo definitivo).

Perché SmolVLM (e non Ollama / CLIP / moondream):
  - È un VLM **generativo** → ragiona sull'immagine e gestisce bene le
    domande con negazione ("è SENZA casco?"), dove CLIP/DINOv2 (similarità
    di embedding) sbagliano.
  - È supportato **nativamente** da `transformers` (niente `trust_remote_code`),
    quindi resta compatibile con le versioni recenti della libreria — a
    differenza di moondream2, il cui codice remoto si rompe con transformers 5.x.
  - Gira **in-process** → niente server esterno, niente Ollama, niente HTTP.
  - Funziona in **zero-shot**: nessun dataset etichettato richiesto.

Modello di default:
  - HuggingFaceTB/SmolVLM-500M-Instruct  (~500M, ~2.5s/query su CPU con
    image-splitting disattivato)

Alternativa più precisa (stessa interfaccia, cambia solo `model_id`):
  - HuggingFaceTB/SmolVLM2-2.2B-Instruct  (più lento su CPU, più accurato, ~9GB di pesi)

Nota performance: la validazione parte SOLO sulle violazioni confermate dal
tracker ed è eseguita fuori dal loop dei frame (vedi VideoSafetyTracker).
Con `do_image_splitting=False` l'immagine usa ~99 token invece di ~900 →
query ~8x più veloce su CPU, a parità di risposta per crop piccoli.

Una domanda per item (non una query multi-item unica): con SmolVLM-500M una
domanda composta su piu' PPE spesso produce una singola risposta secca
("Yes."/"No.") che il parsing per posizione puo' assegnare all'item
sbagliato → falsi negativi (conferma "indossato" quando in realta' manca).
Su un sistema di sicurezza questo e' peggio del costo di N query separate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2 as cv

if TYPE_CHECKING:
    import numpy as np

# Domanda yes/no per singolo PPE. SmolVLM risponde in modo affidabile a
# domande binarie dirette come questa (es. "Yes." / "No.").
#
# Glove/Shoe richiedono ESPLICITAMENTE "on both hands/feet": PersonPPEChecker
# li marca "mancanti" anche quando ne manca solo UNO dei due (REQUIRED_PPE_COUNTS
# = 2). Una domanda generica ("wearing safety gloves?") e' ambigua sul conteggio
# — il VLM puo' vedere un guanto solo su una mano e rispondere comunque "yes",
# scagionando una violazione che invece e' reale (mancanza parziale).
_QUESTION_TEMPLATE = "Is the construction worker in this image wearing {item}? Answer only 'yes' or 'no'."

_PPE_NAMES: dict[str, str] = {
    "Helmet": "a hard hat / safety helmet",
    "Vest": "a high-visibility safety vest",
    "Glove": "safety gloves on BOTH hands",
    "Shoe": "safety boots on BOTH feet",
}


class LocalVLMValidator:
    """
    Valida le violazioni PPE interrogando un piccolo VLM locale (SmolVLM).

    L'inferenza gira in-process tramite ``transformers`` — nessun server
    esterno. Il modello viene caricato in modo pigro alla prima chiamata
    (il primo caricamento scarica i pesi, ~1GB per SmolVLM-500M).

    Parameters
    ----------
    model_id : str
        ID del modello HuggingFace. Default: "HuggingFaceTB/SmolVLM-500M-Instruct".
        Più accurato: "HuggingFaceTB/SmolVLM2-2.2B-Instruct".
    device : str | None
        "cuda", "cpu" o None (auto: cuda se disponibile, altrimenti cpu).
    max_new_tokens : int
        Token generati per risposta. 8 basta per "yes"/"no".
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        device: str | None = None,
        max_new_tokens: int = 8,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        """Messaggio d'errore dell'ultimo `load()` fallito, se presente."""
        return self._load_error

    # ── Public loading ────────────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Carica esplicitamente pesi + processor.

        Da chiamare quando si costruisce il tracker (non durante `run()`),
        cosi' il costo di caricamento (download/lettura pesi da disco) non
        ricade sulla prima violazione confermata a runtime.
        """
        return self._ensure_loaded()

    # ── Availability ──────────────────────────────────────────────────────────
    @staticmethod
    def is_available() -> bool:
        """True se le dipendenze (torch + transformers) sono importabili."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ── Lazy model load ─────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> bool:
        """Carica modello + processor alla prima chiamata. False se fallisce."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            device = self._resolve_device()
            # Su CPU float32; su GPU bfloat16.
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype)
            self._model = model.to(device).eval()
            return True
        except Exception as e:  # noqa: BLE001 — degradare, non crashare la pipeline
            self._load_error = f"{type(e).__name__}: {e}"
            return False

    # ── Inference helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(image_bgr: np.ndarray):
        from PIL import Image

        rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _answer(self, pil, question: str, max_new_tokens: int | None = None) -> str:
        """Interroga SmolVLM con una domanda su una singola immagine."""
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        # do_image_splitting=False: ~99 token invece di ~900 → ~8x più veloce su CPU.
        inputs = self._processor(
            text=prompt, images=[pil], return_tensors="pt", do_image_splitting=False
        ).to(self._model.device)
        n_prompt = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False
            )

        decoded = self._processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    # ── Public API ────────────────────────────────────────────────────────────
    def validate_missing_ppe(
        self,
        crop: np.ndarray,
        missing_ppe: list[str],
    ) -> tuple[bool, str]:
        """
        Chiede al VLM se ogni PPE marcato "mancante" dalla YOLO è davvero assente.

        Returns
        -------
        tuple[bool, str]
            (at_least_one_confirmed_missing, raw_responses)
            True = il VLM conferma che almeno un PPE è realmente mancante.
        """
        if not self._ensure_loaded():
            return False, f"[VLM non caricato: {self._load_error}]"

        try:
            pil = self._to_pil(crop)
        except Exception as e:  # noqa: BLE001
            return False, f"[VLM image error: {e}]"

        confirmed = False
        responses: list[str] = []

        for item in missing_ppe:
            item_name = _PPE_NAMES.get(item, item.lower())
            question = _QUESTION_TEMPLATE.format(item=item_name)
            try:
                answer = self._answer(pil, question).lower()
                # Mostra la frase completa (non solo la chiave categoria "Glove"/"Shoe")
                # cosi' dal log si vede subito che la domanda era su ENTRAMBI gli
                # elementi ("on BOTH hands/feet"), non su un singolo pezzo isolato.
                responses.append(f"{item}[{item_name}]->'{answer}'")
                # "no" → il PPE NON è indossato → violazione confermata dal VLM.
                first_words = answer.replace(",", " ").replace(".", " ").split()[:3]
                if answer.startswith("no") or "no" in first_words or "not wearing" in answer:
                    confirmed = True
            except Exception as e:  # noqa: BLE001
                responses.append(f"{item}[{item_name}]->[error: {e}]")

        return confirmed, " | ".join(responses)
