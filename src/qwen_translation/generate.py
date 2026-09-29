from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a translation with a saved LoRA adapter.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--source-language", default="English")
    parser.add_argument("--target-language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def generate(args: argparse.Namespace) -> str:
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. Translate the user's text from "
                f"{args.source_language} to {args.target_language}. Return only the translation."
            ),
        },
        {"role": "user", "content": args.text},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True).strip()


def main() -> None:
    args = _parse_args()
    print(generate(args))


if __name__ == "__main__":
    main()
