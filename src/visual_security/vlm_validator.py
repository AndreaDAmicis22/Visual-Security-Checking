"""
VLM Validator — Escalation via local vision LLM (Ollama).

Quando il tracker YOLO+PPEChecker conferma una violazione persistente,
il crop della persona viene inviato a un modello vision locale per
validazione finale (riduce i falsi positivi).

Backend: Ollama (https://ollama.com)
Modelli consigliati (in ordine di leggerezza):
  - moondream      (~1.6B, velocissimo, ottimo per VQA yes/no)
  - minicpm-v      (~3B, buon bilanciamento)
  - llava-phi3     (~3.8B, più preciso)

Setup:
  1. Installa Ollama: https://ollama.com/download
  2. Scarica un modello: ollama pull moondream
  3. Il server parte automaticamente su http://localhost:11434
"""

from __future__ import annotations

import base64
import json
import urllib.request
from typing import TYPE_CHECKING

import cv2 as cv

if TYPE_CHECKING:
    import numpy as np


_VLM_PROMPT_TEMPLATE = "Look at this construction worker. Is the worker wearing a {item}? Answer ONLY 'yes' or 'no'."

_PPE_NAMES: dict[str, str] = {
    "Helmet": "hard hat or safety helmet",
    "Vest": "high-visibility safety vest",
    "Glove": "safety gloves",
    "Shoe": "safety boots or shoes",
}


class OllamaVLMValidator:
    """
    Validates PPE violations by querying a local Ollama vision model.

    Parameters
    ----------
    model : str
        Ollama model name (must support vision). Default: "moondream".
    base_url : str
        Ollama API base URL. Default: "http://localhost:11434".
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "moondream",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is loaded."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                return self.model in models
        except Exception:
            return False

    def query(self, image: np.ndarray, prompt: str) -> str:
        """
        Send an image + prompt to the Ollama vision model.

        Parameters
        ----------
        image : np.ndarray
            BGR image (crop of person).
        prompt : str
            Text question.

        Returns
        -------
        str
            Model response text.
        """
        ok, buf = cv.imencode(".jpg", image)
        if not ok:
            msg = "Failed to encode image as JPEG"
            raise RuntimeError(msg)

        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())

        return result.get("response", "").strip()

    def validate_missing_ppe(
        self,
        crop: np.ndarray,
        missing_ppe: list[str],
    ) -> tuple[bool, str]:
        """
        Ask the VLM about each missing PPE item on a person crop.

        Returns
        -------
        tuple[bool, str]
            (at_least_one_confirmed_missing, raw_responses)
            True means the VLM agrees at least one PPE is actually missing.
        """
        confirmed = False
        responses: list[str] = []

        for item in missing_ppe:
            item_name = _PPE_NAMES.get(item, item.lower())
            prompt = _VLM_PROMPT_TEMPLATE.format(item=item_name)

            try:
                answer = self.query(crop, prompt).lower()
                responses.append(f"{item}→'{answer}'")

                # If VLM says "no" → the PPE is NOT worn → violation confirmed
                if answer.startswith("no") or "no" in answer.split()[:3]:
                    confirmed = True
            except Exception as e:
                responses.append(f"{item}→[error: {e}]")

        return confirmed, " | ".join(responses)
