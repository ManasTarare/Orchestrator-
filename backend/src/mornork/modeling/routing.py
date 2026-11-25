"""Inference wrapper for trained discrete route classifiers."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LocalRouteModel:
    def __init__(self, checkpoint: str | Path, *, device: str = "auto", dtype: str = "float16") -> None:
        self.checkpoint = Path(checkpoint)
        metadata_path = self.checkpoint / "mornork-model.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("Orchestrator checkpoint is missing mornork-model.json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.labels = list(self.metadata.get("route_labels") or [])
        if not self.labels:
            raise ValueError("Orchestrator checkpoint has no route_labels")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtypes.get(dtype, torch.float16 if device.startswith("cuda") else torch.float32)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for orchestrator inference but is unavailable")
        self.device = device
        self._preferred_device = device
        self._lock = threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if self.metadata.get("approach") == "fine_tune":
            from peft import PeftModel
            base = AutoModelForSequenceClassification.from_pretrained(
                self.metadata["base_model"], torch_dtype=torch_dtype,
                num_labels=len(self.labels), ignore_mismatched_sizes=True,
                id2label={index: label for index, label in enumerate(self.labels)},
                label2id={label: index for index, label in enumerate(self.labels)},
            )
            self.model = PeftModel.from_pretrained(base, self.checkpoint)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, torch_dtype=torch_dtype)
        self.model.to(device).eval()

    def offload_for_training(self) -> bool:
        if not self.device.startswith("cuda"):
            return False
        with self._lock:
            if not self.device.startswith("cuda"):
                return False
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self.device = "cpu"
            return True

    def restore_after_training(self) -> None:
        if self.device == "cpu" and self._preferred_device.startswith("cuda"):
            with self._lock:
                if self.device == "cpu":
                    self.model.to(self._preferred_device)
                    self.device = self._preferred_device

    @torch.inference_mode()
    def predict(self, message: str) -> tuple[str, float]:
        with self._lock:
            inputs = self.tokenizer(message, return_tensors="pt", truncation=True,
                                    max_length=int(self.metadata.get("sequence_length", 256))).to(self.device)
            probabilities = torch.softmax(self.model(**inputs).logits.float(), dim=-1)[0]
            index = int(probabilities.argmax().item())
            return self.labels[index], float(probabilities[index].item())
