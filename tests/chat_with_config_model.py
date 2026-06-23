from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the LLM configured in tensor_llm_adapter_pipeline.yaml and run a simple chat smoke test."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tensor_llm_adapter_pipeline.yaml",
        help="Tensor-LLM pipeline config.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt. If omitted, the script enters interactive mode.",
    )
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens generated per response.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_model_dtype(raw: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if raw == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported torch dtype: {raw}")
    return mapping[raw]


def resolve_model_path(config: dict[str, Any]) -> str:
    local_dir = first_nested(config, ["model.local_dir"])
    if local_dir:
        resolved = resolve_path_string(local_dir, PROJECT_ROOT)
        if resolved and Path(resolved).exists():
            return resolved
    model_name = first_nested(config, ["model.name_or_path", "model.model_name_or_path"])
    if not model_name:
        raise ValueError("Config must set model.local_dir or model.name_or_path.")
    return str(model_name)


def apply_hf_environment(config: dict[str, Any]) -> tuple[str | None, str | None]:
    hf_home = first_nested(config, ["storage.hf_home"])
    cache_dir = first_nested(config, ["model.cache_dir", "storage.hf_home"])
    resolved_hf_home = resolve_path_string(hf_home, PROJECT_ROOT) if hf_home else None
    resolved_cache_dir = resolve_path_string(cache_dir, PROJECT_ROOT) if cache_dir else None
    if resolved_hf_home:
        os.environ.setdefault("HF_HOME", resolved_hf_home)
    if resolved_cache_dir:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(resolved_cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", resolved_cache_dir)
    return resolved_hf_home, resolved_cache_dir


def build_inputs(tokenizer, prompt: str, system_prompt: str, device: torch.device) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        text = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return input_ids.to(device)


@torch.no_grad()
def generate_reply(
    tokenizer,
    model,
    prompt: str,
    system_prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    input_ids = build_inputs(tokenizer, prompt=prompt, system_prompt=system_prompt, device=device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    config = load_yaml_mapping(args.config)
    _hf_home, cache_dir = apply_hf_environment(config)
    model_name_or_path = resolve_model_path(config)
    device = resolve_device(args.device or first_nested(config, ["runtime.device"], default="auto"))
    torch_dtype = resolve_model_dtype(str(first_nested(config, ["model.torch_dtype"], default="auto")), device)
    trust_remote_code = bool(first_nested(config, ["model.trust_remote_code"], default=False))

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers before running this chat smoke test.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_name_or_path}")
    print(f"Device: {device}; dtype: {str(torch_dtype).replace('torch.', '')}")

    if args.prompt:
        print(generate_reply(
            tokenizer=tokenizer,
            model=model,
            prompt=args.prompt,
            system_prompt=args.system,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        ))
        return

    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        prompt = input("\nUser> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        reply = generate_reply(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            system_prompt=args.system,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        print(f"Assistant> {reply}")


if __name__ == "__main__":
    main()
