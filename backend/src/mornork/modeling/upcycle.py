"""Build and reload a Qwen3-1.7B sparse-MoE checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .moe import Top2MoEMLP

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
ARTIFACT_FORMAT_VERSION = 1


def _qwen_layers(model: torch.nn.Module) -> Any:
    qwen_body = getattr(model, "model", None)
    layers = getattr(qwen_body, "layers", None)
    if layers is None:
        raise TypeError("Expected a Qwen3 causal language model exposing model.layers")
    return layers


def upcycle_model(
    model: torch.nn.Module,
    *,
    base_model_id: str = DEFAULT_BASE_MODEL,
    num_experts: int = 4,
    top_k: int = 2,
    every_n_layers: int = 2,
    first_moe_layer: int = 1,
    noise_std: float = 0.0,
    base_model_revision: str | None = None,
) -> torch.nn.Module:
    """Replace every Nth Qwen3 MLP with a sparse top-k MoE MLP."""
    if every_n_layers < 1:
        raise ValueError("every_n_layers must be at least 1")
    if first_moe_layer < 0:
        raise ValueError("first_moe_layer cannot be negative")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    if num_experts < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("Require num_experts >= 1 and 1 <= top_k <= num_experts")

    layers = _qwen_layers(model)
    layer_indices = list(range(first_moe_layer, len(layers), every_n_layers))
    if not layer_indices:
        raise ValueError("The selected model has no layers to convert")

    hidden_size = int(model.config.hidden_size)
    for index in layer_indices:
        layer = layers[index]
        base_mlp = getattr(layer, "mlp", None)
        if base_mlp is None:
            raise TypeError(f"Qwen3 layer {index} does not expose an mlp module")
        layer.mlp = Top2MoEMLP(
            base_mlp,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=noise_std,
        )

    model.config.mornork_moe = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "num_experts": num_experts,
        "top_k": top_k,
        "every_n_layers": every_n_layers,
        "first_moe_layer": first_moe_layer,
        "noise_std": noise_std,
        "layer_indices": layer_indices,
    }
    return model


def _write_runtime_snapshot(output_dir: Path) -> None:
    package_dir = Path(__file__).resolve().parents[1]
    runtime_package = output_dir / "runtime" / "mornork"
    runtime_modeling = runtime_package / "modeling"
    runtime_modeling.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_dir / "__init__.py", runtime_package / "__init__.py")
    for filename in ("__init__.py", "moe.py", "upcycle.py"):
        shutil.copy2(package_dir / "modeling" / filename, runtime_modeling / filename)


def _resolve_base_license(
    base_model_id: str,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    local_model_dir = Path(base_model_id)
    local_license = local_model_dir / "LICENSE"
    if local_model_dir.is_dir():
        if local_license.is_file():
            return local_license
        raise FileNotFoundError(f"Local base model has no LICENSE file: {local_model_dir}")
    return Path(
        hf_hub_download(
            repo_id=base_model_id,
            filename="LICENSE",
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def save_upcycled_model(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    base_license_file: str | Path | None = None,
    max_shard_size: str = "4GB",
) -> Path:
    """Save weights, config, metadata, and the loader source used by this project."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    moe_config = getattr(model.config, "mornork_moe", None)
    if not moe_config:
        raise ValueError("Model has no Mornork MoE configuration; call upcycle_model first")

    model.save_pretrained(target, safe_serialization=True, max_shard_size=max_shard_size)
    if tokenizer is not None:
        tokenizer.save_pretrained(target)

    license_relative_path = None
    if base_license_file is not None:
        license_dir = target / "THIRD_PARTY_LICENSES"
        license_dir.mkdir(parents=True, exist_ok=True)
        license_name = Path(moe_config["base_model_id"]).name + "-LICENSE"
        license_target = license_dir / license_name
        shutil.copy2(base_license_file, license_target)
        license_relative_path = license_target.relative_to(target).as_posix()

    _write_runtime_snapshot(target)
    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_name": "Mornork",
        "base_model": moe_config["base_model_id"],
        "base_model_revision": moe_config.get("base_model_revision"),
        "moe": moe_config,
        "tokenizer": "saved with the model" if tokenizer is not None else None,
        "license_file": license_relative_path,
        "loader": "mornork.modeling.upcycle.load_upcycled_model",
    }
    (target / "mornork-artifact.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _checkpoint_shards(checkpoint_dir: Path) -> list[Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
        return [checkpoint_dir / name for name in shard_names]

    single_file = checkpoint_dir / "model.safetensors"
    if single_file.is_file():
        return [single_file]
    raise FileNotFoundError(
        f"No safetensors checkpoint found in {checkpoint_dir}; "
        "Mornork checkpoints must use safe_serialization"
    )


def _load_safetensors_into_model(
    model: torch.nn.Module,
    checkpoint_dir: Path,
) -> None:
    state = model.state_dict(keep_vars=True)
    loaded: set[str] = set()
    unexpected: list[str] = []

    with torch.no_grad():
        for shard_path in _checkpoint_shards(checkpoint_dir):
            if not shard_path.is_file():
                raise FileNotFoundError(f"Checkpoint shard is missing: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    target = state.get(name)
                    if target is None:
                        unexpected.append(name)
                        continue
                    tensor = shard.get_tensor(name)
                    if target.shape != tensor.shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, "
                            f"model {tuple(target.shape)}"
                        )
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    loaded.add(name)

    if unexpected:
        raise ValueError(f"Unexpected checkpoint weights: {unexpected[:8]}")

    missing = set(state) - loaded
    # Safe serialization omits duplicate entries for tied weights. A missing
    # key is valid only when it aliases a parameter that was loaded by another key.
    loaded_storage = {
        (state[name].device, state[name].data_ptr(), tuple(state[name].shape))
        for name in loaded
    }
    missing_not_tied = []
    for name in missing:
        target = state[name]
        storage = (target.device, target.data_ptr(), tuple(target.shape))
        if storage not in loaded_storage:
            missing_not_tied.append(name)
    if missing_not_tied:
        raise ValueError(f"Checkpoint is missing weights: {missing_not_tied[:8]}")


def load_upcycled_model(
    checkpoint_dir: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Recreate Mornork's custom layers, then load its safetensors weights."""
    source = Path(checkpoint_dir)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    moe_config = getattr(config, "mornork_moe", None)
    if not moe_config:
        raise ValueError(f"{source} is not a Mornork upcycled checkpoint")

    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    upcycle_model(
        model,
        base_model_id=moe_config["base_model_id"],
        base_model_revision=moe_config.get("base_model_revision"),
        num_experts=int(moe_config["num_experts"]),
        top_k=int(moe_config["top_k"]),
        every_n_layers=int(moe_config["every_n_layers"]),
        first_moe_layer=int(moe_config["first_moe_layer"]),
        noise_std=float(moe_config.get("noise_std", 0.0)),
    )
    _load_safetensors_into_model(model, source)
    if device is not None:
        model.to(device)
    return model


def _dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upcycle Qwen3-1.7B into Mornork's initial sparse-MoE form."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--revision", help="Optional pinned model revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--every-n-layers", type=int, default=2)
    parser.add_argument("--first-moe-layer", type=int, default=1)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Keep at zero for the release checkpoint; tune only for fine-tuning.",
    )
    parser.add_argument(
        "--device-map",
        help="Optional Transformers device map, such as auto (requires accelerate).",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    load_kwargs: dict[str, Any] = {
        "torch_dtype": _dtype_from_name(args.dtype),
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "use_safetensors": True,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    base_license_file = _resolve_base_license(
        args.base_model,
        args.revision,
        args.local_files_only,
    )
    upcycle_model(
        model,
        base_model_id=args.base_model,
        base_model_revision=args.revision,
        num_experts=args.num_experts,
        top_k=args.top_k,
        every_n_layers=args.every_n_layers,
        first_moe_layer=args.first_moe_layer,
        noise_std=args.noise_std,
    )
    destination = save_upcycled_model(
        model,
        args.output_dir,
        tokenizer=tokenizer,
        base_license_file=base_license_file,
        max_shard_size=args.max_shard_size,
    )
    print(f"Saved Mornork checkpoint to {destination}")


if __name__ == "__main__":
    main()
