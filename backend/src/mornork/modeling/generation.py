"""Local text generation using a saved Mornork checkpoint."""

from __future__ import annotations

import threading
import re
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .upcycle import load_upcycled_model


def _drop_incomplete_tail(text: str) -> str:
    """Avoid presenting an unfinished final sentence when generation hits its token cap."""
    value = text.rstrip()
    if re.search(r"[.!?](?:[\"'’”»)}\]]|\*\*)*\s*$", value):
        return value
    boundaries = list(re.finditer(r"[.!?](?=(?:[\"'’”»)}\]]|\*\*)*(?:\s|$))", value))
    if not boundaries:
        return "I reached the response length limit before completing the answer. Please ask for a shorter version."
    end = boundaries[-1].end()
    while end < len(value) and value[end] in "\"'’”»)}] ":
        end += 1
    if value[end:end + 2] == "**":
        end += 2
    return value[:end].rstrip()


class LocalChatModel:
    """Load a local upcycled checkpoint and generate one deterministic reply at a time."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 160,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(f"Mornork checkpoint directory not found: {self.checkpoint}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for generation, but PyTorch reports CUDA unavailable")
        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            torch_dtype = dtypes[dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported generation dtype: {dtype}") from error

        self.device = device
        self._preferred_device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            local_files_only=True,
        )
        model_manifest = self.checkpoint / "mornork-model.json"
        if model_manifest.is_file():
            metadata = json.loads(model_manifest.read_text(encoding="utf-8"))
            if metadata.get("approach") == "fine_tune":
                try:
                    from peft import PeftModel
                except ImportError as error:
                    raise RuntimeError("This checkpoint uses LoRA adapters; install the training extra") from error
                base_model = metadata.get("base_model")
                if not base_model:
                    raise ValueError("Fine-tuned checkpoint is missing its base_model metadata")
                base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
                self.model = PeftModel.from_pretrained(base, self.checkpoint)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype=torch_dtype)
            self.model.to(device)
        else:
            self.model = load_upcycled_model(
                self.checkpoint,
                dtype=torch_dtype,
                device=device,
            )
        self.model = self.model.eval()
        self._generation_lock = threading.Lock()

    def offload_for_training(self) -> bool:
        """Move the default model off the GPU before a separate training process starts."""
        if not self.device.startswith("cuda"):
            return False
        with self._generation_lock:
            if not self.device.startswith("cuda"):
                return False
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self.device = "cpu"
            return True

    def restore_after_training(self) -> None:
        """Return the default model to its configured device after training ends."""
        if self.device == "cpu" and self._preferred_device.startswith("cuda"):
            with self._generation_lock:
                if self.device == "cpu":
                    self.model.to(self._preferred_device)
                    self.device = self._preferred_device

    def generate(self, message: str, *, system_prompt: str | None = None) -> str:
        if not message.strip():
            raise ValueError("message cannot be blank")
        conversation = []
        response_style = (
            "Keep the answer concise: use a few sentences or at most five short bullets. "
            "Finish every sentence. Do not start a new section or bullet unless you can complete it."
        )
        prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else ""
        system_content = f"{prompt}\n\n{response_style}" if prompt else response_style
        conversation.append({"role": "system", "content": system_content})
        conversation.append({"role": "user", "content": message})
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        else:
            plain_prompt = f"### System\n{system_content}\n\n### User\n{message.strip()}\n\n### Assistant\n"
            encoded = self.tokenizer(plain_prompt, return_tensors="pt", add_special_tokens=True)
        if isinstance(encoded, Mapping):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        elif hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        else:
            input_ids = encoded
            attention_mask = None
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generation_args = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if attention_mask is not None:
            generation_args["attention_mask"] = attention_mask

        with self._generation_lock, torch.inference_mode():
            output = self.model.generate(**generation_args)
        generated_tokens = output.shape[-1] - input_ids.shape[-1]
        answer = self.tokenizer.decode(
            output[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        if generated_tokens >= self.max_new_tokens:
            answer = _drop_incomplete_tail(answer)
        if not answer:
            raise RuntimeError("Mornork generated an empty response")
        return answer
