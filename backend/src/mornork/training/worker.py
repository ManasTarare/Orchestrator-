"""Run one local causal-language-model training job in its own process."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tokenizers import ByteLevelBPETokenizer

from .artifacts import ModelArtifactStore
from .data import deterministic_route_splits, load_qa_examples, load_route_examples
from .metrics import calculate_answer_metrics
from .store import TrainingStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_matches(row: dict[str, Any], split: str | None) -> bool:
    if split is None:
        return True
    values = {str(key).strip().casefold(): value for key, value in row.items()}
    if "split" not in values:
        return False
    value = str(values.get("split") or "").strip().casefold()
    if split == "validation":
        return value in {"validation", "valid", "val"}
    return value == split


def _load_texts(dataset: dict[str, Any], split: str | None = None) -> list[str]:
    path = Path(dataset["file_path"])
    suffix = path.suffix.casefold()
    texts: list[str] = []
    if suffix == ".txt":
        texts = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            has_split = any(str(key).strip().casefold() == "split" for key in (reader.fieldnames or []))
            for row in reader:
                if not split or not has_split or _split_matches(row, split):
                    texts.append(_row_to_text(row))
    elif suffix == ".jsonl":
        dataset_has_split = any(str(column).strip().casefold() == "split" for column in dataset.get("columns", []))
        with path.open("r", encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL on line {line_no}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} must contain an object")
                if not split or not dataset_has_split or _split_matches(row, split):
                    texts.append(_row_to_text(row))
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("PDF training requires the training extra: python -m pip install -e .[training]") from error
        reader = PdfReader(str(path))
        texts = [(page.extract_text() or "").strip() for page in reader.pages]
    else:
        raise ValueError(f"Unsupported training data file type: {suffix or 'unknown'}")
    # Stable de-duplication keeps repeated rows from overweighting a tiny local corpus.
    return list(dict.fromkeys(text.strip() for text in texts if text.strip()))


def _row_to_text(row: dict[str, Any]) -> str:
    row = {str(key).strip().casefold(): value for key, value in row.items()}
    if "approved" in row and str(row["approved"]).strip().casefold() not in {"true", "yes", "1", "approved"}:
        return ""
    def value(*names: str) -> str:
        for name in names:
            item = row.get(name)
            if item is not None and str(item).strip():
                return str(item).strip()
        return ""

    plain = value("text", "content", "document", "article")
    if plain:
        return plain
    prompt = value("prompt", "question", "instruction", "input")
    response = value("response", "answer", "output", "completion")
    context = value("context", "source", "reference")
    if prompt and response:
        context_parts = [context] if context else []
        excerpt = value("official_source_excerpt", "source_excerpt", "excerpt")
        period = value("tax_period", "tax_year", "financial_year", "effective_period")
        section = value("source_section", "section", "paragraph")
        source_url = value("source_url", "url", "citation_url")
        if period:
            context_parts.append(f"Applicable period: {period}")
        if section:
            context_parts.append(f"Official source section: {section}")
        if excerpt:
            context_parts.append(f"Official source excerpt: {excerpt}")
        if source_url:
            context_parts.append(f"Source URL: {source_url}")
        middle = f"\nContext: {'; '.join(context_parts)}" if context_parts else ""
        return f"### User\n{prompt}{middle}\n\n### Assistant\n{response}"
    return ""


class TokenBlockDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, block_size: int) -> None:
        self.blocks: list[list[int]] = []
        for text in texts:
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if not encoded:
                continue
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                encoded = [*encoded, int(eos_id)]
            for offset in range(0, len(encoded), block_size):
                block = encoded[offset : offset + block_size]
                if len(block) >= 2:
                    self.blocks.append(block)
        if not self.blocks:
            raise ValueError("No training sequences were produced; provide longer text or more rows")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"input_ids": self.blocks[index]}


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
        try:
            encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
            if encoded is None:
                raise ValueError("Tokenizer chat template did not return input_ids")
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if isinstance(encoded, (list, tuple)) and encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(token) for token in encoded]
    pieces = []
    for message in messages:
        role = message["role"].capitalize()
        pieces.append(f"### {role}\n{message['content']}\n")
    if add_generation_prompt:
        pieces.append("### Assistant\n")
    return [int(token) for token in tokenizer.encode("\n".join(pieces), add_special_tokens=True)]


class AnswerOnlyQADataset(Dataset):
    """Mask prompt loss and fit answers to the token window without changing prompts."""

    def __init__(self, examples: list[dict[str, str]], tokenizer: Any, max_length: int,
                 system_prompt: str, max_prompt_length: int | None = None,
                 *, truncate_answers: bool = True, evaluation_only: bool = False) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.prompt_ids: list[list[int]] = []
        self.examples: list[dict[str, str]] = []
        self.skipped_prompts = 0
        self.truncated_responses = 0
        self.example_count = len(examples)
        self.evaluation_only = evaluation_only
        if max_length < 16:
            raise ValueError("QA sequence length must be at least 16 tokens")
        prompt_limit = min(max_length - 1, max_prompt_length or max_length - 1)
        eos_id = getattr(tokenizer, "eos_token_id", None)

        for index, example in enumerate(examples, start=1):
            answer = example["expected"].strip()
            if not answer:
                continue
            response_messages = [{"role": "assistant", "content": answer}]

            def encode(context: str) -> tuple[list[int], list[int]]:
                user = example["prompt"].strip()
                if context.strip():
                    user += "\n\nContext:\n" + context.strip()
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ]
                if getattr(tokenizer, "chat_template", None):
                    prefix = _chat_ids(tokenizer, conversation, add_generation_prompt=True)
                    if self.evaluation_only:
                        return prefix, []
                    full = _chat_ids(tokenizer, conversation + response_messages, add_generation_prompt=False)
                    if full[:len(prefix)] != prefix:
                        raise ValueError("Tokenizer chat template does not keep the generation prompt as a prefix")
                    target = full[len(prefix):]
                else:
                    prefix_text = (f"### System\n{system_prompt}\n\n### User\n{user}\n\n### Assistant\n")
                    prefix = [int(token) for token in tokenizer.encode(prefix_text, add_special_tokens=True)]
                    if self.evaluation_only:
                        return prefix, []
                    target = [int(token) for token in tokenizer.encode(answer, add_special_tokens=False)]
                    if eos_id is not None:
                        target.append(int(eos_id))
                if not target and eos_id is not None:
                    target = [int(eos_id)]
                return prefix, target

            context = example.get("context", "")
            prefix, target = encode(context)
            # Always preserve the question. Training/validation answers may be
            # shortened to the available window; held-out test references remain
            # complete so the reported task metrics score the original targets.
            if len(prefix) > prompt_limit:
                self.skipped_prompts += 1
                continue
            if self.evaluation_only:
                self.examples.append(example)
                self.prompt_ids.append(prefix)
                continue
            available_answer_tokens = max_length - len(prefix)
            if truncate_answers and len(target) > available_answer_tokens:
                target = target[:available_answer_tokens]
                if target and eos_id is not None:
                    target[-1] = int(eos_id)
                self.truncated_responses += 1
            if not target:
                self.skipped_prompts += 1
                continue

            input_ids = prefix + target
            labels = ([-100] * len(prefix)) + target.copy()
            self.examples.append(example)
            self.prompt_ids.append(prefix)
            self.items.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
        if not self.prompt_ids:
            raise ValueError(
                f"No usable QA prompts fit sequence_length={max_length}; "
                "select a longer sequence length or use shorter questions"
            )

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class AnswerOnlyCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id or 0

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        ids, masks, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [int(self.pad_token_id)] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def _training_overall_progress(state: Any, epochs: int) -> float:
    """Estimate run progress across training and each epoch's validation."""
    epochs = max(1, epochs)
    maximum = max(int(getattr(state, "max_steps", 0) or 0), 1)
    step = min(maximum, max(0, int(getattr(state, "global_step", 0))))
    if step == 0:
        return 15.0
    steps_per_epoch = max(1, math.ceil(maximum / epochs))
    epoch_index = min(epochs - 1, (step - 1) // steps_per_epoch)
    fraction = ((step - 1) % steps_per_epoch + 1) / steps_per_epoch
    return 15.0 + (70.0 * epoch_index + 60.0 * fraction) / epochs


def _validation_overall_progress(state: Any, epochs: int, completed: int, total: int) -> float:
    """Map held-out validation batches into their share of total run progress."""
    epochs = max(1, epochs)
    epoch_number = int(round(float(getattr(state, "epoch", 0.0) or 0.0)))
    epoch_index = min(epochs - 1, max(0, epoch_number - 1))
    fraction = min(1.0, max(0.0, completed / max(1, total)))
    validation_start = 15.0 + (70.0 * epoch_index + 60.0) / epochs
    return validation_start + 10.0 * fraction / epochs


class RunProgress(TrainerCallback):
    def __init__(self, store: TrainingStore, job_id: str, epochs: int) -> None:
        self.store = store
        self.job_id = job_id
        self.epochs = max(1, epochs)
        self.stage = "training"

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any):
        values = logs or {}
        maximum = max(int(getattr(state, "max_steps", 0) or 0), 1)
        metrics = {key: value for key, value in values.items() if isinstance(value, (int, float))}
        step = int(getattr(state, "global_step", 0))
        testing = any(key.startswith("test_") for key in values)
        evaluating = testing or any(key.startswith("eval_") for key in values)
        progress = (96.0 if testing else _validation_overall_progress(state, self.epochs, 1, 1) if evaluating
                    else _training_overall_progress(state, self.epochs))
        self.store.add_event(
            self.job_id,
            level="INFO",
            stage="evaluation" if testing else "validation" if evaluating else self.stage,
            progress=progress,
            message=("Held-out test-set language-model metrics" if testing else
                     f"Held-out validation metrics at step {step} of {maximum}" if evaluating
                     else f"Training step {step} of {maximum}"),
            metrics=metrics,
        )
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
        if self.store.is_cancel_requested(self.job_id):
            control.should_training_stop = True
        return control


class EvaluationProgress(TrainerCallback):
    """Publish validation progress while preserving epoch-level run progress."""

    def __init__(self, store: TrainingStore, job_id: str, total_batches: int, epochs: int) -> None:
        self.store = store
        self.job_id = job_id
        self.total_batches = max(1, total_batches)
        self.epochs = max(1, epochs)
        self.completed = 0
        self.last_reported = 0
        self.enabled = True

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any):
        self.completed = 0
        self.last_reported = 0
        return control

    def on_prediction_step(self, args: Any, state: Any, control: Any, **kwargs: Any):
        if not self.enabled:
            return control
        self.completed += 1
        interval = max(1, self.total_batches // 25)
        if self.completed == self.total_batches or self.completed - self.last_reported >= interval:
            self.last_reported = self.completed
            progress = _validation_overall_progress(state, self.epochs, self.completed, self.total_batches)
            self.store.add_event(
                self.job_id, level="INFO", stage="validation", progress=progress,
                message=f"Held-out evaluation batch {self.completed} of {self.total_batches}",
                metrics={"evaluation_batches": self.completed, "evaluation_total_batches": self.total_batches},
            )
        return control


class TaskEvaluationCallback(TrainerCallback):
    """Evaluate one deterministic sample of the held-out test split after training."""

    def __init__(self, store: TrainingStore, job_id: str, *, model: Any, tokenizer: Any,
                 qa_dataset: AnswerOnlyQADataset | None = None,
                 qa_available_count: int | None = None,
                 route_labels: list[str] | None = None,
                 route_test_rows: list[dict[str, str]] | None = None,
                 route_sequence_length: int = 128, epochs: int = 1,
                 max_new_tokens: int = 64, qa_sample_size: int = 200) -> None:
        self.store = store
        self.job_id = job_id
        self.model = model
        self.tokenizer = tokenizer
        self.qa_dataset = qa_dataset
        self.qa_available_count = qa_available_count
        self.route_labels = route_labels
        self.route_test_rows = route_test_rows or []
        self.route_sequence_length = max(1, int(route_sequence_length))
        self.epochs = max(1, epochs)
        self.max_new_tokens = max_new_tokens
        self.qa_sample_size = max(1, qa_sample_size)
        self.metrics: dict[str, Any] | None = None
        self.cases: list[dict[str, Any]] = []

    def on_evaluate(self, args: Any, state: Any, control: Any,
                    metrics: dict[str, Any] | None = None, **kwargs: Any):
        epoch = int(round(float(getattr(state, "epoch", 0.0) or 0.0)))
        final_epoch = epoch >= self.epochs
        if self.route_labels is not None:
            if final_epoch and self.route_test_rows:
                self._evaluate_routes()
            return control

        if self.qa_dataset is None:
            return control
        if not final_epoch:
            return control
        previous_training = bool(self.model.training)
        previous_cache = getattr(self.model.config, "use_cache", None)
        self.model.eval()
        self.model.config.use_cache = True
        device = next(self.model.parameters()).device
        generated: list[dict[str, Any]] = []
        dataset_count = len(self.qa_dataset)
        available_count = max(dataset_count, int(self.qa_available_count or dataset_count))
        sample_count = min(dataset_count, self.qa_sample_size)
        sample_indices = sorted(random.Random(42).sample(range(dataset_count), sample_count))
        interval = max(1, sample_count // 25)
        self.store.add_event(
            self.job_id, level="INFO", stage="evaluation", progress=85.0,
            message=f"Scoring a deterministic sample of {sample_count} held-out answers",
            metrics={"task_evaluation_examples": 0, "task_evaluation_total": sample_count,
                     "task_evaluation_available": available_count},
        )
        try:
            with torch.inference_mode():
                for sample_index, dataset_index in enumerate(sample_indices, start=1):
                    if self.store.is_cancel_requested(self.job_id):
                        return control
                    example = self.qa_dataset.examples[dataset_index]
                    prompt_ids = self.qa_dataset.prompt_ids[dataset_index]
                    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    max_positions = (getattr(self.model.config, "max_position_embeddings", None)
                                     or getattr(self.model.config, "n_positions", None))
                    available_new_tokens = self.max_new_tokens
                    if max_positions:
                        available_new_tokens = min(available_new_tokens, max(1, int(max_positions) - len(prompt_ids)))
                    output = self.model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=available_new_tokens, do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    actual = self.tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
                    generated.append({"case_id": sample_index, "prompt": example["prompt"],
                                      "context": example.get("context", ""),
                                      "expected": example["expected"], "actual": actual})
                    if sample_index == sample_count or sample_index % interval == 0:
                        fraction = sample_index / max(1, sample_count)
                        progress = 85.0 + 10.0 * fraction
                        self.store.add_event(
                            self.job_id, level="INFO", stage="evaluation", progress=progress,
                            message=f"Scoring test answers {sample_index} of {sample_count}",
                            metrics={"task_evaluation_examples": sample_index, "task_evaluation_total": sample_count,
                                     "task_evaluation_available": available_count},
                        )
        finally:
            if previous_cache is not None:
                self.model.config.use_cache = previous_cache
            if previous_training:
                self.model.train()

        self.cases = generated
        self.metrics = calculate_answer_metrics(generated, "test", sample_count, available_count)
        self.store.add_event(
            self.job_id, level="INFO", stage="evaluation", progress=96.0,
            message=f"Test metrics calculated on {len(generated)} held-out answers",
            metrics={"task_metrics": self.metrics},
        )
        return control

    def _evaluate_routes(self) -> None:
        available_count = len(self.route_test_rows)
        sample_count = min(available_count, self.qa_sample_size)
        indices = sorted(random.Random(42).sample(range(available_count), sample_count))
        selected = [self.route_test_rows[index] for index in indices]
        label_to_id = {label: index for index, label in enumerate(self.route_labels or [])}
        device = next(self.model.parameters()).device
        was_training = bool(self.model.training)
        self.model.eval()
        predicted_ids: list[int] = []
        expected_ids: list[int] = []
        self.store.add_event(self.job_id, level="INFO", stage="evaluation", progress=85.0,
                             message=f"Scoring {sample_count} held-out test routes",
                             metrics={"task_evaluation_examples": 0, "task_evaluation_total": sample_count,
                                      "task_evaluation_available": available_count})
        try:
            with torch.inference_mode():
                for offset in range(0, sample_count, 4):
                    rows = selected[offset:offset + 4]
                    encoded = self.tokenizer(
                        [row["message"] for row in rows], truncation=True,
                        max_length=self.route_sequence_length, padding=True, return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    batch_predictions = self.model(**encoded).logits.argmax(dim=-1).detach().cpu().tolist()
                    predicted_ids.extend(int(value) for value in batch_predictions)
                    expected_ids.extend(label_to_id[row["route"]] for row in rows)
                    completed = len(predicted_ids)
                    self.store.add_event(
                        self.job_id, level="INFO", stage="evaluation",
                        progress=85.0 + 10.0 * completed / max(1, sample_count),
                        message=f"Scoring test routes {completed} of {sample_count}",
                        metrics={"task_evaluation_examples": completed,
                                 "task_evaluation_total": sample_count,
                                 "task_evaluation_available": available_count},
                    )
                    for row, predicted_id in zip(rows, batch_predictions):
                        self.cases.append({"case_id": len(self.cases) + 1,
                                           "prompt": row["message"], "context": "",
                                           "expected": row["route"],
                                           "actual": self.route_labels[int(predicted_id)]})
        finally:
            if was_training:
                self.model.train()

        metrics = _route_metrics((
            torch.nn.functional.one_hot(torch.tensor(predicted_ids), num_classes=len(self.route_labels or [])).numpy(),
            torch.tensor(expected_ids).numpy(),
        ), self.route_labels)
        confusion = json.loads(metrics.pop("confusion_matrix_json"))
        self.metrics = {
            "metric_version": 2, "task_type": "route_classification", "split": "test",
            "examples_evaluated": sample_count, "available_examples": available_count,
            "requested_cases": sample_count, **metrics,
            "confusion_matrix": {"labels": confusion["labels"], "matrix": confusion["matrix"],
                                 "row_meaning": "actual", "column_meaning": "predicted"},
            "confusion_matrix_note": "Rows are actual labels; columns are predicted labels.",
        }
        self.store.add_event(self.job_id, level="INFO", stage="evaluation", progress=96.0,
                             message=f"Test route metrics calculated on {sample_count} held-out messages",
                             metrics={"task_metrics": self.metrics})


def _train_tokenizer(texts: list[str], output_dir: Path, vocab_size: int) -> PreTrainedTokenizerFast:
    special = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(iter(texts), vocab_size=vocab_size, min_frequency=2, special_tokens=special)
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )
    wrapped.save_pretrained(output_dir / "tokenizer")
    return wrapped


def _training_arguments(output_dir: Path, config: dict[str, Any], evaluation: bool) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    use_cuda = torch.cuda.is_available() and str(config.get("device", "auto")) != "cpu"
    values: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config.get("epochs", 1)),
        "per_device_train_batch_size": int(config.get("batch_size", 1)),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(config.get("learning_rate", 3e-4)),
        "weight_decay": 0.01,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": use_cuda,
        "optim": "adamw_torch",
        "fp16": use_cuda,
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        "seed": int(config.get("seed", 42)),
    }
    strategy = "epoch" if evaluation else "no"
    if "eval_strategy" in parameters:
        values["eval_strategy"] = strategy
    elif "evaluation_strategy" in parameters:
        values["evaluation_strategy"] = strategy
    if "use_cpu" in parameters:
        values["use_cpu"] = str(config.get("device", "auto")) == "cpu"
    elif "no_cuda" in parameters:
        values["no_cuda"] = str(config.get("device", "auto")) == "cpu"
    # TrainingArguments forwards **kwargs internally on some Transformers releases,
    # even when a kwarg is not actually supported. Only pass explicit parameters.
    values = {name: value for name, value in values.items() if name in parameters}
    return TrainingArguments(**values)


class RouteClassificationDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]], tokenizer: Any, labels: list[str], max_length: int) -> None:
        label_ids = {label: index for index, label in enumerate(labels)}
        self.items = []
        for example in examples:
            encoded = tokenizer(example["message"], truncation=True, max_length=max_length)
            encoded["labels"] = label_ids[example["route"]]
            self.items.append(encoded)
        if not self.items:
            raise ValueError("No orchestrator examples were produced")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _route_metrics(eval_prediction: Any, route_labels: list[str] | None = None) -> dict[str, Any]:
    logits, expected = eval_prediction
    predicted = logits.argmax(axis=-1).tolist()
    expected_values = expected.tolist()
    label_ids = list(range(len(route_labels))) if route_labels else sorted(set(expected_values) | set(predicted))
    accuracy = sum(int(left == right) for left, right in zip(expected_values, predicted)) / max(1, len(expected_values))
    precisions, recalls, f1s = [], [], []
    for label in label_ids:
        tp = sum(1 for truth, guess in zip(expected_values, predicted) if truth == label and guess == label)
        fp = sum(1 for truth, guess in zip(expected_values, predicted) if truth != label and guess == label)
        fn = sum(1 for truth, guess in zip(expected_values, predicted) if truth == label and guess != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(precision); recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    matrix_labels = list(range(len(route_labels or label_ids)))
    matrix = [[0 for _ in matrix_labels] for _ in matrix_labels]
    for truth, guess in zip(expected_values, predicted):
        if truth in matrix_labels and guess in matrix_labels:
            matrix[truth][guess] += 1
    names = route_labels or [str(label) for label in matrix_labels]
    return {"accuracy": accuracy, "macro_precision": sum(precisions) / max(1, len(precisions)),
            "macro_recall": sum(recalls) / max(1, len(recalls)), "macro_f1": sum(f1s) / max(1, len(f1s)),
            "confusion_matrix_json": json.dumps({"labels": names, "matrix": matrix})}


def _explicit_split(dataset: dict[str, Any] | None) -> bool:
    return bool(dataset and any(str(column).strip().casefold() == "split" for column in dataset.get("columns", [])))


def _prepare_expert_text_splits(job: dict[str, Any], primary: dict[str, Any],
                                evaluation: dict[str, Any] | None) -> tuple[list[str], list[str], list[str]]:
    seed = int(job["config"].get("seed", 42))
    if evaluation is not None:
        if _explicit_split(primary):
            train = _load_texts(primary, "train")
            validation = _load_texts(primary, "validation")
        else:
            texts = _load_texts(primary)
            indices = list(range(len(texts)))
            random.Random(seed).shuffle(indices)
            if len(indices) < 3:
                raise ValueError("Training data needs at least 3 records to create train/validation plus a separate test set")
            train_end = max(1, int(len(indices) * 0.8))
            validation_count = max(1, int(len(indices) * 0.1))
            if train_end + validation_count >= len(indices):
                train_end, validation_count = len(indices) - 2, 1
            train = [texts[index] for index in indices[:train_end]]
            validation = [texts[index] for index in indices[train_end:train_end + validation_count]]
        if _explicit_split(evaluation):
            validation = validation or _load_texts(evaluation, "validation")
            test = _load_texts(evaluation, "test") or _load_texts(primary, "test")
        else:
            test = _load_texts(evaluation)
        if _explicit_split(evaluation) and not test:
            raise ValueError("Separate evaluation data needs split=test records")
        return train, validation, test
    if _explicit_split(primary):
        train = _load_texts(primary, "train")
        validation = _load_texts(primary, "validation")
        test = _load_texts(primary, "test")
        if not validation:
            raise ValueError("Use split=validation for the training validation partition; keep split=test reserved for final metrics")
        if not test:
            raise ValueError("Text training needs held-out split=test records for automatic in-run evaluation")
        return train, validation, test
    texts = _load_texts(primary)
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    if len(indices) < 3:
        raise ValueError("Use at least 3 records to create disjoint train, validation, and test splits")
    train_end = max(1, int(len(indices) * 0.8))
    validation_count = max(1, int(len(indices) * 0.1))
    if train_end + validation_count >= len(indices):
        train_end, validation_count = len(indices) - 2, 1
    train = [texts[i] for i in indices[:train_end]]
    validation = [texts[i] for i in indices[train_end:train_end + validation_count]]
    test = [texts[i] for i in indices[train_end + validation_count:]]
    return train, validation, test


def _has_qa_format(dataset: dict[str, Any]) -> bool:
    columns = {str(column).strip().casefold() for column in dataset.get("columns", [])}
    return bool(columns & {"prompt", "question", "instruction", "input"}) and bool(
        columns & {"response", "answer", "output", "completion"})


def _prepare_expert_qa_splits(job: dict[str, Any], primary: dict[str, Any],
                              evaluation: dict[str, Any] | None
                              ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    seed = int(job["config"].get("seed", 42))
    if _explicit_split(primary):
        train = load_qa_examples(primary, "train")
        validation = load_qa_examples(primary, "validation")
        test = load_qa_examples(primary, "test")
    else:
        examples = load_qa_examples(primary)
        if len(examples) < 3:
            raise ValueError("QA training data needs at least 3 approved examples for train/validation/test")
        indices = list(range(len(examples)))
        random.Random(seed).shuffle(indices)
        train_end = max(1, int(len(indices) * 0.8))
        validation_count = max(1, int(len(indices) * 0.1))
        if train_end + validation_count >= len(indices):
            train_end, validation_count = len(indices) - 2, 1
        validation_end = train_end + validation_count
        train = [examples[index] for index in indices[:train_end]]
        validation = [examples[index] for index in indices[train_end:validation_end]]
        test = [examples[index] for index in indices[validation_end:]]

    if evaluation is not None:
        if _explicit_split(evaluation):
            validation = validation or load_qa_examples(evaluation, "validation")
            test = load_qa_examples(evaluation, "test") or test
            if not test:
                raise ValueError("Separate QA evaluation data needs split=test records")
        else:
            test = load_qa_examples(evaluation)
    if not train:
        raise ValueError("The QA training split contains no approved prompt/response examples")
    if not validation:
        raise ValueError("QA training needs held-out split=validation examples for automatic in-run metrics")
    if not test:
        raise ValueError("QA training needs held-out split=test examples for automatic in-run evaluation")
    return train, validation, test


def _prepare_route_splits(job: dict[str, Any], primary: dict[str, Any],
                          evaluation: dict[str, Any] | None) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    seed = int(job["config"].get("seed", 42))
    if _explicit_split(primary):
        train = load_route_examples(primary, "train")
        validation = load_route_examples(primary, "validation")
        test = load_route_examples(primary, "test")
        if evaluation is not None:
            if _explicit_split(evaluation):
                validation = validation or load_route_examples(evaluation, "validation")
                test = test or load_route_examples(evaluation, "test")
            elif not test:
                test = load_route_examples(evaluation)
        if not validation:
            raise ValueError("Orchestrator data needs held-out split=validation examples")
        if not test:
            raise ValueError("Orchestrator data needs held-out split=test examples or a separate test dataset")
    else:
        all_examples = load_route_examples(primary)
        if evaluation is None:
            train, validation, test = deterministic_route_splits(all_examples, seed)
        else:
            train, validation, _unused_test = deterministic_route_splits(all_examples, seed)
            test = load_route_examples(evaluation, "test") if _explicit_split(evaluation) else load_route_examples(evaluation)
            if not test:
                raise ValueError("The separate orchestrator evaluation file has no usable test rows")
    if len(train) < 2 or len({example["route"] for example in train}) < 2:
        raise ValueError("Orchestrator training data needs at least two route classes and two approved training rows")
    labels = {example["route"] for example in train}
    unseen = {example["route"] for example in validation + test} - labels
    if unseen:
        raise ValueError(f"Held-out data contains route labels absent from training: {', '.join(sorted(unseen))}")
    return train, validation, test


def _make_trainer(model: Any, tokenizer: Any, train_set: Dataset, eval_set: Dataset,
                  data_collator: Any, output_dir: Path, config: dict[str, Any],
                  store: TrainingStore, job_id: str,
                  compute_metrics: Any = None,
                  task_evaluation: TaskEvaluationCallback | None = None
                  ) -> tuple[Trainer, RunProgress, EvaluationProgress]:
    # Evaluation batch size is explicitly one in _training_arguments.
    eval_batches = max(1, len(eval_set))
    epochs = int(config.get("epochs", 1))
    progress = RunProgress(store, job_id, epochs)
    evaluation_progress = EvaluationProgress(store, job_id, eval_batches, epochs)
    arguments = _training_arguments(output_dir / "trainer", config, evaluation=True)
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    kwargs: dict[str, Any] = {
        "model": model, "args": arguments, "train_dataset": train_set, "eval_dataset": eval_set,
        "data_collator": data_collator,
        "callbacks": [progress, evaluation_progress] + ([task_evaluation] if task_evaluation else []),
    }
    if compute_metrics:
        kwargs["compute_metrics"] = compute_metrics
    if "processing_class" in trainer_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**kwargs)
    trainer.mornork_task_evaluation = task_evaluation
    return trainer, progress, evaluation_progress


def _train_expert(job: dict[str, Any], train_texts: list[str], validation_texts: list[str],
                  output_dir: Path, model_dir: Path, store: TrainingStore,
                  train_qa: list[dict[str, str]] | None = None,
                  validation_qa: list[dict[str, str]] | None = None,
                  test_qa: list[dict[str, str]] | None = None,
                  ) -> tuple[Any, Any, str, Trainer, RunProgress, EvaluationProgress, dict[str, Any]]:
    config = job["config"]
    block_size = int(config.get("sequence_length", 128))
    if job["approach"] == "scratch":
        tokenizer_texts = train_texts
        if not tokenizer_texts and train_qa:
            tokenizer_texts = [
                f"### System\n{job.get('description', '')}\n\n### User\n{item['prompt']}\n\nContext:\n{item.get('context', '')}\n\n### Assistant\n{item['expected']}"
                for item in train_qa
            ]
        store.add_event(job["id"], level="INFO", stage="tokenizer", progress=12,
                        message="Training a new byte-level BPE tokenizer using training rows only")
        tokenizer = _train_tokenizer(tokenizer_texts, output_dir, int(config.get("vocab_size", 4096)))
        hidden, heads = int(config.get("hidden_size", 128)), int(config.get("attention_heads", 4))
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        store.add_event(job["id"], level="INFO", stage="model_initialization", progress=14,
                        message="Initializing a Transformers GPT-2-style Transformer from random weights")
        model_config = GPT2Config(vocab_size=len(tokenizer), n_positions=block_size, n_ctx=block_size,
                                  n_embd=hidden, n_layer=int(config.get("layers", 2)), n_head=heads,
                                  n_inner=hidden * 4, bos_token_id=tokenizer.bos_token_id,
                                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                  use_cache=False)
        model = GPT2LMHeadModel(model_config)
        model_type = "gpt2"
    else:
        base_model = str(job.get("base_model") or "").strip()
        if not base_model:
            raise ValueError("Choose a base model for fine-tuning")
        store.add_event(job["id"], level="INFO", stage="model_initialization", progress=12,
                        message=f"Loading pretrained base model {base_model} for LoRA fine-tuning")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model_dtype = torch.float16 if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=model_dtype, low_cpu_mem_usage=True)
        model_type = getattr(model.config, "model_type", "transformers")
        model_limit = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
        if model_limit:
            block_size = min(block_size, int(model_limit))
        model.config.use_cache = False
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise RuntimeError("LoRA fine-tuning requires the training extra: python -m pip install -e .[training]") from error
        desired = {"c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "query", "key", "value", "dense"}
        found = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        target_modules = sorted(desired & found) or "all-linear"
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                                lora_dropout=0.05, target_modules=target_modules))
    task_evaluation = None
    if train_qa is not None and validation_qa is not None:
        system_prompt = str(job.get("description") or "").strip() or (
            "Answer the question accurately using only the supplied context. "
            "If it is insufficient, say so. Keep the answer concise."
        )
        model_limit = (getattr(model.config, "max_position_embeddings", None)
                       or getattr(model.config, "n_positions", None))
        prompt_limit = min(block_size, int(model_limit)) if model_limit else block_size
        train_set = AnswerOnlyQADataset(train_qa, tokenizer, block_size, system_prompt, prompt_limit)
        eval_set = AnswerOnlyQADataset(validation_qa, tokenizer, block_size, system_prompt, prompt_limit)
        all_test_qa = test_qa or []
        # Keep the whole held-out split available for deterministic sampling. Only
        # prompts must fit the input window; references stay complete for scoring.
        test_set = AnswerOnlyQADataset(
            all_test_qa, tokenizer, block_size, system_prompt, prompt_limit,
            truncate_answers=False, evaluation_only=True,
        )
        collator = AnswerOnlyCollator(tokenizer)
        task_evaluation = TaskEvaluationCallback(
            store, job["id"], model=model, tokenizer=tokenizer, qa_dataset=test_set,
            qa_available_count=len(all_test_qa),
            epochs=int(config.get("epochs", 1)),
        )
        if (train_set.skipped_prompts or eval_set.skipped_prompts or test_set.skipped_prompts
                or train_set.truncated_responses or eval_set.truncated_responses):
            store.add_event(job["id"], level="WARN", stage="preparing_data", progress=13,
                            message=(f"Trimmed answers in {train_set.truncated_responses} training and "
                                     f"{eval_set.truncated_responses} validation rows to sequence_length={block_size}; "
                                     f"excluded {train_set.skipped_prompts} training, {eval_set.skipped_prompts} validation, "
                                     f"and {test_set.skipped_prompts} test rows whose prompts alone exceed the window. "
                                     "Test reference answers were kept complete for scoring."))
    else:
        train_set = TokenBlockDataset(train_texts, tokenizer, block_size)
        eval_set = TokenBlockDataset(validation_texts, tokenizer, block_size)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    store.add_event(job["id"], level="INFO", stage="training", progress=15,
                    message=(f"Starting {job['approach']} expert answer-model training on {len(train_set)} train and "
                             f"{len(eval_set)} held-out validation examples"))
    trainer, progress, evaluation_progress = _make_trainer(
        model, tokenizer, train_set, eval_set, collator,
        output_dir, config, store, job["id"], task_evaluation=task_evaluation,
    )
    return model, tokenizer, model_type, trainer, progress, evaluation_progress, {"sequence_length": block_size}


def _train_orchestrator(job: dict[str, Any], train_rows: list[dict[str, str]],
                        validation_rows: list[dict[str, str]], test_rows: list[dict[str, str]],
                        output_dir: Path, model_dir: Path,
                        store: TrainingStore) -> tuple[Any, Any, str, Trainer, RunProgress, EvaluationProgress, dict[str, Any]]:
    config = job["config"]
    labels = sorted({row["route"] for row in train_rows})
    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    sequence_length = int(config.get("sequence_length", 128))
    if job["approach"] == "scratch":
        store.add_event(job["id"], level="INFO", stage="tokenizer", progress=12,
                        message="Training a routing tokenizer from approved training messages only")
        tokenizer = _train_tokenizer([row["message"] for row in train_rows], output_dir,
                                     int(config.get("vocab_size", 4096)))
        hidden, heads = int(config.get("hidden_size", 128)), int(config.get("attention_heads", 4))
        if hidden % heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        model_config = GPT2Config(vocab_size=len(tokenizer), n_positions=sequence_length,
                                  n_ctx=sequence_length, n_embd=hidden, n_layer=int(config.get("layers", 2)),
                                  n_head=heads, n_inner=hidden * 4, num_labels=len(labels),
                                  id2label=id2label, label2id=label2id,
                                  pad_token_id=tokenizer.pad_token_id, use_cache=False)
        model = GPT2ForSequenceClassification(model_config)
        model_type = "gpt2_sequence_classifier"
        store.add_event(job["id"], level="INFO", stage="model_initialization", progress=14,
                        message="Initializing a Transformers route-classification Transformer from random weights")
    else:
        base_model = str(job.get("base_model") or "").strip()
        if not base_model:
            raise ValueError("Choose a base model for orchestrator fine-tuning")
        store.add_event(job["id"], level="INFO", stage="model_initialization", progress=12,
                        message=f"Loading pretrained sequence-classification base {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model_dtype = torch.float16 if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(labels), id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True, torch_dtype=model_dtype,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model_type = getattr(model.config, "model_type", "transformers") + "_sequence_classifier"
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise RuntimeError("Orchestrator LoRA fine-tuning requires python -m pip install -e .[training]") from error
        desired = {"c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "query", "key", "value", "dense"}
        found = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        head = "score" if "score" in module_names else "classifier" if "classifier" in module_names else None
        modules_to_save = [head] if head else None
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                                lora_dropout=0.05, target_modules=sorted(desired & found) or "all-linear",
                                                modules_to_save=modules_to_save))
        max_positions = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None)
        if max_positions:
            sequence_length = min(sequence_length, int(max_positions))
    train_set = RouteClassificationDataset(train_rows, tokenizer, labels, sequence_length)
    eval_set = RouteClassificationDataset(validation_rows, tokenizer, labels, sequence_length)
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    store.add_event(job["id"], level="INFO", stage="training", progress=15,
                    message=f"Training route classifier with {len(labels)} route labels; {len(train_set)} train and {len(eval_set)} validation examples")
    trainer, progress, evaluation_progress = _make_trainer(
        model, tokenizer, train_set, eval_set, DataCollatorWithPadding(tokenizer=tokenizer),
        output_dir, config, store, job["id"],
        compute_metrics=lambda prediction: _route_metrics(prediction, labels),
        task_evaluation=TaskEvaluationCallback(
            store, job["id"], model=model, tokenizer=tokenizer,
            route_labels=labels, route_test_rows=test_rows, route_sequence_length=sequence_length,
            epochs=int(config.get("epochs", 1)),
        ),
    )
    return model, tokenizer, model_type, trainer, progress, evaluation_progress, {
        "sequence_length": sequence_length, "route_labels": labels, "label2id": label2id, "id2label": id2label,
    }


def run_job(store: TrainingStore, job_id: str) -> None:
    job = store.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if store.is_cancel_requested(job_id):
        store.update_job(job_id, status="cancelled", stage="cancelled", finished_at=_now())
        store.add_event(job_id, level="WARN", stage="cancelled", progress=0, message="Run cancelled before training started")
        return
    store.update_job(job_id, status="running", stage="preparing_data", started_at=_now(), worker_pid=os.getpid())
    store.add_event(job_id, level="INFO", stage="preparing_data", progress=2, message="Reading and validating this model path's data format")
    primary = store.get_dataset(job["dataset_id"])
    evaluation_record = store.get_dataset(job["eval_dataset_id"]) if job.get("eval_dataset_id") else None
    if primary is None:
        raise ValueError("Training dataset was removed")
    if job["target"] == "orchestrator":
        train_rows, validation_rows, test_rows = _prepare_route_splits(job, primary, evaluation_record)
        train_texts = validation_texts = []
        train_qa = validation_qa = []
        split_counts = {"train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)}
    else:
        train_rows = validation_rows = test_rows = []
        if _has_qa_format(primary):
            train_qa, validation_qa, test_qa = _prepare_expert_qa_splits(job, primary, evaluation_record)
            train_texts = validation_texts = []
            split_counts = {"train": len(train_qa), "validation": len(validation_qa), "test": len(test_qa)}
        else:
            train_qa = validation_qa = []
            train_texts, validation_texts, test_texts = _prepare_expert_text_splits(job, primary, evaluation_record)
            split_counts = {"train": len(train_texts), "validation": len(validation_texts), "test": len(test_texts)}
    expert_train_data = train_qa if train_qa else train_texts
    expert_validation_data = validation_qa if validation_qa else validation_texts
    if not (train_rows if job["target"] == "orchestrator" else expert_train_data):
        raise ValueError("The training split contains no usable approved examples")
    if not (validation_rows if job["target"] == "orchestrator" else expert_validation_data):
        raise ValueError("A non-empty held-out validation split is required")
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = job["config"]
    store.add_event(job_id, level="INFO", stage="preparing_data", progress=10,
                    message=f"Prepared disjoint splits: {split_counts['train']} train, {split_counts['validation']} validation, {split_counts['test']} test")
    if job["target"] == "orchestrator":
        model, tokenizer, model_type, trainer, progress_callback, eval_callback, task_metadata = _train_orchestrator(
            job, train_rows, validation_rows, test_rows, output_dir, model_dir, store)
        model_task = "route_classifier"
    else:
        model, tokenizer, model_type, trainer, progress_callback, eval_callback, task_metadata = _train_expert(
            job, train_texts, validation_texts, output_dir, model_dir, store,
            train_qa if train_qa else None, validation_qa if validation_qa else None,
            test_qa if train_qa else None)
        model_task = "answer_generation"
    resume = any((output_dir / "trainer").glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    if store.is_cancel_requested(job_id):
        store.update_job(job_id, status="cancelled", stage="cancelled", progress=100, finished_at=_now(), worker_pid=None)
        store.add_event(job_id, level="WARN", stage="cancelled", progress=100, message="Training run cancelled by user")
        return
    test_language_metrics: dict[str, Any] = {}
    if job["target"] == "expert" and not train_qa and test_texts:
        test_set = TokenBlockDataset(test_texts, tokenizer, int(task_metadata["sequence_length"]))
        available_blocks = len(test_set)
        sample_count = min(available_blocks, 200)
        selected_blocks = sorted(random.Random(42).sample(range(available_blocks), sample_count))
        interval = max(1, sample_count // 25)
        progress_callback.set_stage("evaluation")
        store.add_event(job_id, level="INFO", stage="evaluation", progress=85,
                        message=f"Scoring {sample_count} held-out test text blocks",
                        metrics={"task_evaluation_examples": 0,
                                 "task_evaluation_total": sample_count,
                                 "task_evaluation_available": available_blocks})
        was_training = bool(model.training)
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        try:
            with torch.inference_mode():
                for index, block_index in enumerate(selected_blocks, start=1):
                    if store.is_cancel_requested(job_id):
                        store.update_job(job_id, status="cancelled", stage="cancelled", progress=100,
                                         finished_at=_now(), worker_pid=None)
                        store.add_event(job_id, level="WARN", stage="cancelled", progress=100,
                                        message="Training run cancelled during test evaluation")
                        return
                    batch = collator([test_set[block_index]])
                    batch = {key: value.to(device) for key, value in batch.items()}
                    output = model(**batch)
                    target_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                    total_loss += float(output.loss.detach().item()) * max(1, target_tokens)
                    total_tokens += target_tokens
                    if index == sample_count or index % interval == 0:
                        store.add_event(
                            job_id, level="INFO", stage="evaluation",
                            progress=85.0 + 10.0 * index / max(1, sample_count),
                            message=f"Scoring test text blocks {index} of {sample_count}",
                            metrics={"task_evaluation_examples": index,
                                     "task_evaluation_total": sample_count,
                                     "task_evaluation_available": available_blocks},
                        )
        finally:
            if was_training:
                model.train()
        test_loss = total_loss / max(1, total_tokens)
        test_language_metrics = {
            "test_loss": test_loss,
            "test_perplexity": math.exp(min(test_loss, 80)),
            "test_examples_evaluated": sample_count,
            "test_available_examples": available_blocks,
            "test_split": "test",
        }
        store.add_event(job_id, level="INFO", stage="evaluation", progress=96,
                        message=f"Test language-model metrics calculated on {sample_count} text blocks",
                        metrics=test_language_metrics)
    last_eval = next((entry for entry in reversed(trainer.state.log_history)
                      if any(key.startswith("eval_") for key in entry)), {})
    eval_loss = float(last_eval["eval_loss"]) if "eval_loss" in last_eval else None
    perplexity = math.exp(min(eval_loss, 80)) if eval_loss is not None else None
    validation_task_metrics = {key.removeprefix("eval_"): float(value) for key, value in last_eval.items()
                               if key.startswith("eval_") and key not in {"eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "eval_steps"}
                               and isinstance(value, (int, float))}
    task_evaluation = getattr(trainer, "mornork_task_evaluation", None)
    task_metrics = task_evaluation.metrics if task_evaluation else None
    if task_metrics:
        validation_task_metrics["task_metrics"] = task_metrics
    progress_callback.set_stage("saving")
    store.add_event(job_id, level="INFO", stage="saving", progress=96,
                    message="Saving tokenizer, model configuration, and version manifest")
    tokenizer.save_pretrained(model_dir)
    model.config.use_cache = True if model_task == "answer_generation" else getattr(model.config, "use_cache", False)
    model.save_pretrained(model_dir, safe_serialization=True)
    store.add_event(job_id, level="INFO", stage="saving", progress=98,
                    message="Model weights and tokenizer are saved locally")
    manifest = {
        "job_id": job_id, "model_name": job["model_name"], "target": job["target"], "domain": job["domain"],
        "approach": job["approach"], "architecture": job["architecture"], "model_type": model_type,
        "model_task": model_task, "base_model": job.get("base_model"), "dataset_id": job["dataset_id"],
        "eval_dataset_id": job.get("eval_dataset_id"), "split_counts": split_counts,
        "evaluation": {"eval_loss": eval_loss, "perplexity": perplexity, **test_language_metrics,
                       "task_metrics": task_metrics, **{key: value for key, value in validation_task_metrics.items()
                                                        if key != "task_metrics"}},
        "readiness": ("evaluated_review_required" if test_language_metrics
                      or (task_metrics and task_metrics.get("split") == "test")
                      else "evaluation_required"),
        "created_at": _now(), **task_metadata,
    }
    (model_dir / "mornork-model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    version = store.save_model_version(job, model_dir, manifest["evaluation"])
    manifest.update({"id": version["id"], "version": version["version"],
                     "readiness": version["readiness"]})
    (model_dir / "mornork-model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if task_evaluation and task_metrics and task_evaluation.cases:
        store.create_evaluation(version["id"], task_evaluation.cases)
    artifacts = ModelArtifactStore()
    artifact_uri = None
    if artifacts.upload_enabled:
        store.add_event(job_id, level="INFO", stage="saving", progress=98,
                        message="Uploading versioned model artifacts to configured S3 storage")
        try:
            artifact_uri = artifacts.upload_directory(model_dir, job["target"], version["id"])
            store.update_artifact_uri(version["id"], artifact_uri)
            store.add_event(job_id, level="INFO", stage="saving", progress=99,
                            message=f"S3 artifact saved: {artifact_uri}")
        except Exception as error:
            store.add_event(job_id, level="WARN", stage="saving", progress=98,
                            message=f"S3 upload failed; local checkpoint was kept ({type(error).__name__}: {error})")
    store.update_job(job_id, status="built_evaluated", stage="complete", progress=100,
                     finished_at=_now(), worker_pid=None)
    store.add_event(job_id, level="INFO", stage="complete", progress=100,
                    message="Checkpoint saved with held-out test metrics from this run; no second evaluation is needed.",
                    metrics={"eval_loss": eval_loss, "perplexity": perplexity,
                             **test_language_metrics,
                             "task_metrics": task_metrics,
                             "artifact_uri": artifact_uri})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    store = TrainingStore(args.database, args.datasets, args.artifacts)
    try:
        run_job(store, args.job_id)
    except Exception as error:
        job = store.get_job(args.job_id)
        if job and job["status"] not in {"cancelled", "built_evaluation_required", "built_evaluated"}:
            message = f"{type(error).__name__}: {error}"
            store.update_job(args.job_id, status="failed", stage="failed", error_message=message,
                             finished_at=_now(), worker_pid=None)
            store.add_event(args.job_id, level="ERROR", stage="failed", progress=None,
                            message=message[:1000])
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
