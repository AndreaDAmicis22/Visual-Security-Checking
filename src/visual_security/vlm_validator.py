"""
VLM Validator — Escalation via a local, in-process vision LLM.

Quando il tracker YOLO+PPEChecker conferma una violazione persistente,
il crop della persona viene inviato a un piccolo VLM *locale* per una
seconda opinione (riduce i falsi positivi di una YOLO ancora non
addestrata in modo definitivo).

Perché moondream2 e non Ollama/CLIP:
  - È un VLM **generativo** → ragiona sull'immagine e gestisce bene le
    domande con negazione ("è SENZA casco?"), dove CLIP/DINOv2 (similarità
    di embedding) sbagliano.
  - Gira **in-process** via `transformers` → niente server esterno, niente
    Ollama, niente HTTP/JSON fragile.
  - Funziona in **zero-shot**: nessun dataset etichettato richiesto.

Modello di default:
  - vikhyatk/moondream2  (~2B, VQA, CPU-friendly ma lento su CPU)

Alternativa più leggera (cambia solo `model_id`, stessa interfaccia):
  - HuggingFaceTB/SmolVLM-500M-Instruct  (più veloce su CPU, meno preciso)

Nota performance: su CPU una query può richiedere alcuni secondi. Per questo
la validazione parte SOLO sulle violazioni confermate dal tracker e viene
eseguita fuori dal loop dei frame (vedi VideoSafetyTracker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2 as cv

if TYPE_CHECKING:
    import numpy as np

# Domanda yes/no per singolo PPE. moondream2 risponde in modo affidabile
# a domande binarie e diritte come questa.
_QUESTION_TEMPLATE = "Is the construction worker in this image wearing {item}? Answer only 'yes' or 'no'."

_PPE_NAMES: dict[str, str] = {
    "Helmet": "a hard hat / safety helmet",
    "Vest": "a high-visibility safety vest",
    "Glove": "safety gloves",
    "Shoe": "safety boots",
}


class LocalVLMValidator:
    """
    Valida le violazioni PPE interrogando un piccolo VLM locale (moondream2).

    L'inferenza gira in-process tramite ``transformers`` — nessun server
    esterno. Il modello viene caricato in modo pigro alla prima chiamata
    (il primo caricamento scarica i pesi, ~4GB per moondream2).

    Parameters
    ----------
    model_id : str
        ID del modello HuggingFace. Default: "vikhyatk/moondream2".
        Alternativa leggera: "HuggingFaceTB/SmolVLM-500M-Instruct".
    revision : str | None
        Revisione/tag del modello (moondream2 usa tag datati). Ignorata
        per modelli che non la richiedono.
    device : str | None
        "cuda", "cpu" o None (auto: cuda se disponibile, altrimenti cpu).
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str | None = "2025-06-21",
        device: str | None = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self._model = None
        self._load_error: str | None = None

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
        """Carica il modello alla prima chiamata. Ritorna False se fallisce."""
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM

            device = self._resolve_device()
            # Su CPU float32 è più stabile/veloce di bf16; su GPU bf16.
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
            if self.revision and "moondream" in self.model_id:
                kwargs["revision"] = self.revision

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            model = model.to(device)
            model.eval()
            self._model = model
            return True
        except Exception as e:  # noqa: BLE001 — vogliamo degradare, non crashare la pipeline
            self._load_error = f"{type(e).__name__}: {e}"
            return False

    # ── Inference helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(image_bgr: np.ndarray):
        from PIL import Image

        rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _answer(self, encoded_or_image, question: str) -> str:
        """Interroga il modello. Supporta moondream (.query) come default."""
        # moondream2 espone .query(image_or_encoded, question) -> {"answer": ...}
        out = self._model.query(encoded_or_image, question)
        return (out.get("answer") if isinstance(out, dict) else str(out)).strip()

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
            # Codifica l'immagine una volta sola, poi riusa per ogni domanda.
            image_ref = self._model.encode_image(pil) if hasattr(self._model, "encode_image") else pil
        except Exception as e:  # noqa: BLE001
            return False, f"[VLM encode error: {e}]"

        confirmed = False
        responses: list[str] = []

        for item in missing_ppe:
            item_name = _PPE_NAMES.get(item, item.lower())
            question = _QUESTION_TEMPLATE.format(item=item_name)
            try:
                answer = self._answer(image_ref, question).lower()
                responses.append(f"{item}→'{answer}'")
                # "no" → il PPE NON è indossato → violazione confermata dal VLM.
                first_words = answer.replace(",", " ").split()[:3]
                if answer.startswith("no") or "no" in first_words or "not wearing" in answer:
                    confirmed = True
            except Exception as e:  # noqa: BLE001
                responses.append(f"{item}→[error: {e}]")

        return confirmed, " | ".join(responses)
