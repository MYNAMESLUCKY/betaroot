# ai_query_system.py
import os
import sys
import time
import json
import queue
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    ViltProcessor,
    ViltForQuestionAnswering,
)

# ---------------------------------------------------------------------------
# Logging & constants
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "ai_query_system.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

HF_API_KEY = os.getenv("HF_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------
class DefenseDatasetRegistry:
    def __init__(self):
        self.datasets = self._load_registry()

    @staticmethod
    def _load_registry() -> Dict[str, Any]:
        sample = {
            "satellite_targets": {
                "description": "Multispectral satellite imagery for tank / aircraft detection.",
                "source": "Kaggle (synthetic placeholder)",
                "size_gb": 8.2,
                "classes": ["tank", "jet", "ship", "radar"],
                "url": "https://www.kaggle.com/satellite-targets",
            },
            "signal_intel": {
                "description": "RF signal snapshots for SIGINT classification.",
                "source": "Internal collection",
                "size_gb": 2.4,
                "signal_types": ["communication", "radar", "jamming", "noise"],
                "url": "https://www.kaggle.com/signal-intelligence",
            },
            "cyber_threats": {
                "description": "Network telemetry for cyber intrusion detection.",
                "source": "US-CERT public feeds",
                "size_gb": 5.7,
                "events": ["anomaly", "intrusion", "ddos", "phishing"],
                "url": "https://www.kaggle.com/cyber-threat-defense",
            },
        }
        logging.info("Loaded %d dataset entries", len(sample))
        return sample

    def list_datasets(self) -> Dict[str, Any]:
        return self.datasets

    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        return self.datasets.get(name)


class HuggingFaceVisualAssistant:
    def __init__(self):
        self.hf_token = HF_API_KEY
        if not self.hf_token:
            logging.warning("HF_API_KEY missing, model calls may fail.")
        self.caption_model = None
        self.caption_processor = None
        self.vqa_model = None
        self.vqa_processor = None
        self._load_models()

    def _load_models(self):
        logging.info("Loading Hugging Face models on %s", DEVICE)
        start = time.time()
        try:
            self.caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large", use_auth_token=self.hf_token
            )
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large", use_auth_token=self.hf_token
            ).to(DEVICE)
            logging.info("Loaded BLIP captioning model.")
        except Exception as exc:
            logging.exception("BLIP load failed: %s", exc)
        try:
            self.vqa_processor = ViltProcessor.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa", use_auth_token=self.hf_token
            )
            self.vqa_model = ViltForQuestionAnswering.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa", use_auth_token=self.hf_token
            ).to(DEVICE)
            logging.info("Loaded ViLT VQA model.")
        except Exception as exc:
            logging.exception("ViLT load failed: %s", exc)
        logging.info("Model load completed in %.1fs", time.time() - start)

    def generate_caption(self, image_path: str) -> str:
        if not self.caption_model or not self.caption_processor:
            logging.warning("Caption model unavailable; returning mock response.")
            return "Satellite image showing high-value assets and mixed infrastructure."
        image = Image.open(image_path).convert("RGB")
        inputs = self.caption_processor(images=image, return_tensors="pt").to(DEVICE)
        output = self.caption_model.generate(**inputs, max_length=64)
        caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    def answer_visual_question(self, image_path: str, question: str) -> str:
        if not self.vqa_model or not self.vqa_processor:
            logging.warning("VQA model unavailable; returning mock answer.")
            return "Likely presence of two armored vehicles near radar arrays."
        image = Image.open(image_path).convert("RGB")
        inputs = self.vqa_processor(
            image, question, return_tensors="pt"
        ).to(DEVICE)
        outputs = self.vqa_model(**inputs)
        idx = outputs.logits.argmax(-1).item()
        return self.vqa_model.config.id2label[idx]


class DefenseAIQuerySystem:
    def __init__(self):
        self.registry = DefenseDatasetRegistry()
        self.hf_assistant = HuggingFaceVisualAssistant()
        self.history: List[Dict[str, Any]] = []
        logging.info("Defense AI Query System initialized.")

    # Dataset QA ----------------------------------------------------------------
    def describe_dataset(self, dataset_name: str) -> str:
        dataset = self.registry.get_dataset(dataset_name)
        if not dataset:
            return f"No dataset named '{dataset_name}' found."
        summary = (
            f"Dataset '{dataset_name}': {dataset['description']}\n"
            f"- Source: {dataset['source']}\n"
            f"- Size: {dataset['size_gb']} GB\n"
            f"- Reference: {dataset['url']}"
        )
        logging.info("Describe dataset: %s", dataset_name)
        return summary

    def recommend_dataset(self, query: str) -> str:
        scored = []
        for name, meta in self.registry.list_datasets().items():
            keywords = json.dumps(meta).lower()
            score = sum(
                1 for term in query.lower().split() if term in keywords
            )
            scored.append((score, name))
        scored.sort(reverse=True)
        best = scored[0]
        if best[0] == 0:
            return "No strong matches; try specifying satellite, signal, or cyber needs."
        return self.describe_dataset(best[1])

    # Visual QA -----------------------------------------------------------------
    def caption_image(self, image_path: str) -> str:
        caption = self.hf_assistant.generate_caption(image_path)
        self._log_event("caption", {"image": image_path, "caption": caption})
        return caption

    def vqa(self, image_path: str, question: str) -> str:
        answer = self.hf_assistant.answer_visual_question(image_path, question)
        self._log_event("vqa", {"image": image_path, "question": question, "answer": answer})
        return answer

    # Natural language QA -------------------------------------------------------
    def answer_text_query(self, query: str) -> str:
        query = query.lower()
        if "list datasets" in query or "what datasets" in query:
            lines = []
            for name, meta in self.registry.list_datasets().items():
                lines.append(f"- {name}: {meta['description']}")
            resp = "\n".join(lines)
        elif "recommend" in query or "best dataset" in query:
            resp = self.recommend_dataset(query)
        elif "signal" in query:
            resp = "Signals dataset contains 4 RF classes and labeled SNR metrics."
        elif "satellite" in query:
            resp = "Satellite targets dataset captures multispectral passes with 10 defense classes."
        else:
            resp = "Query logged—no direct answer. Please refine with dataset names or tasks."
        self._log_event("text_query", {"query": query, "response": resp})
        return resp

    # History -------------------------------------------------------------------
    def _log_event(self, kind: str, payload: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": kind,
            "payload": payload,
        }
        self.history.append(entry)
        logging.info("Event logged: %s", entry)

    def dump_history(self, path: Path = Path("query_history.json")):
        path.write_text(json.dumps(self.history, indent=2))
        logging.info("History written to %s", path)

    # CLI -----------------------------------------------------------------------
    def interactive_loop(self):
        banner = (
            "\n=== Defense AI Query System ===\n"
            "Commands: dataset <name> | recommend <need> | caption <img> | "
            "vqa <img>::<question> | history | quit\n"
        )
        print(banner)
        while True:
            try:
                user_input = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input in {"quit", "exit"}:
                break
            if user_input == "history":
                print(json.dumps(self.history, indent=2))
                continue
            if user_input.startswith("dataset "):
                name = user_input.split(" ", 1)[1]
                print(self.describe_dataset(name))
                continue
            if user_input.startswith("recommend "):
                need = user_input.split(" ", 1)[1]
                print(self.recommend_dataset(need))
                continue
            if user_input.startswith("caption "):
                img = user_input.split(" ", 1)[1]
                print(self.caption_image(img))
                continue
            if user_input.startswith("vqa "):
                payload = user_input.split(" ", 1)[1]
                if "::" not in payload:
                    print("Use format: vqa path::question")
                    continue
                img, question = payload.split("::", 1)
                print(self.vqa(img.strip(), question.strip()))
                continue
            # fallback to NLP answer
            print(self.answer_text_query(user_input))
        self.dump_history()


if __name__ == "__main__":
    system = DefenseAIQuerySystem()
    system.interactive_loop()
