from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import sacrebleu
from datasets import load_dataset
from peft import PeftModel
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a base Qwen model with a LoRA adapter.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset-name", default="Helsinki-NLP/opus-100")
    parser.add_argument("--dataset-config", default="en-zh")
    parser.add_argument("--split", default="test")
    parser.add_argument("--source-field", default="en")
    parser.add_argument("--target-field", default="zh")
    parser.add_argument("--source-language", default="English")
    parser.add_argument("--target-language", default="Chinese")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200, help="0 means the full split.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples-in-latex", type=int, default=12)
    return parser.parse_args()


def field_value(example: dict[str, Any], field: str) -> str:
    if field in example:
        value = example[field]
    elif isinstance(example.get("translation"), dict) and field in example["translation"]:
        value = example["translation"][field]
    else:
        raise KeyError(f"Cannot find {field!r} in dataset fields {sorted(example)}")
    return str(value).strip()


def normalize_chinese(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def chinese_char_spaced(text: str) -> str:
    return " ".join(normalize_chinese(text))


def system_prompt(source_language: str, target_language: str) -> str:
    return (
        "You are a professional translator. Translate the user's text from "
        f"{source_language} to {target_language}. Return only the translation."
    )


def conversations(sources: list[str], source_language: str, target_language: str):
    prompt = system_prompt(source_language, target_language)
    return [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": source},
        ]
        for source in sources
    ]


def apply_template(tokenizer, messages):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "padding": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(base_model: str, adapter: str):
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype if use_cuda else torch.float32,
    }
    if use_cuda:
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return tokenizer, model


def model_device(model):
    return model.get_input_embeddings().weight.device


def generate_batch(tokenizer, model, sources: list[str], args: argparse.Namespace, adapter_enabled: bool) -> list[str]:
    inputs = apply_template(
        tokenizer,
        conversations(sources, args.source_language, args.target_language),
    )
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    else:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    device = model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    prompt_length = input_ids.shape[-1]
    return [
        tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def select_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        cache_dir=args.dataset_cache_dir,
    )
    indices = list(range(len(raw)))
    random.Random(args.seed).shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]
    examples = []
    for index in indices:
        row = raw[int(index)]
        examples.append(
            {
                "dataset_index": int(index),
                "source": field_value(row, args.source_field),
                "reference": field_value(row, args.target_field),
            }
        )
    return examples


def score_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    references = [chinese_char_spaced(row["reference"]) for row in rows]
    base_predictions = [chinese_char_spaced(row["base_prediction"]) for row in rows]
    lora_predictions = [chinese_char_spaced(row["lora_prediction"]) for row in rows]
    base_bleu = corpus_bleu(base_predictions, [references], tokenize="none")
    lora_bleu = corpus_bleu(lora_predictions, [references], tokenize="none")
    base_chrf = corpus_chrf(
        [normalize_chinese(row["base_prediction"]) for row in rows],
        [[normalize_chinese(row["reference"]) for row in rows]],
    )
    lora_chrf = corpus_chrf(
        [normalize_chinese(row["lora_prediction"]) for row in rows],
        [[normalize_chinese(row["reference"]) for row in rows]],
    )

    for row in rows:
        reference = chinese_char_spaced(row["reference"])
        base = chinese_char_spaced(row["base_prediction"])
        lora = chinese_char_spaced(row["lora_prediction"])
        base_sentence_chrf = sentence_chrf(normalize_chinese(row["base_prediction"]), [normalize_chinese(row["reference"]) ]).score
        lora_sentence_chrf = sentence_chrf(normalize_chinese(row["lora_prediction"]), [normalize_chinese(row["reference"]) ]).score
        base_sentence_bleu = sentence_bleu(base, [reference], tokenize="none").score
        lora_sentence_bleu = sentence_bleu(lora, [reference], tokenize="none").score
        row.update(
            {
                "base_bleu": round(base_sentence_bleu, 6),
                "lora_bleu": round(lora_sentence_bleu, 6),
                "base_chrf": round(base_sentence_chrf, 6),
                "lora_chrf": round(lora_sentence_chrf, 6),
                "chrf_delta": round(lora_sentence_chrf - base_sentence_chrf, 6),
                "base_chars": len(normalize_chinese(row["base_prediction"])),
                "lora_chars": len(normalize_chinese(row["lora_prediction"])),
                "reference_chars": len(normalize_chinese(row["reference"])),
                "changed": row["base_prediction"].strip() != row["lora_prediction"].strip(),
            }
        )

    wins = sum(row["chrf_delta"] > 1e-9 for row in rows)
    losses = sum(row["chrf_delta"] < -1e-9 for row in rows)
    ties = len(rows) - wins - losses
    deltas = [row["chrf_delta"] for row in rows]
    rng = random.Random(42)
    bootstrap_means = []
    for _ in range(1000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        bootstrap_means.append(sum(sample) / len(sample))
    bootstrap_means.sort()
    summary = {
        "examples": len(rows),
        "base_bleu_char": round(base_bleu.score, 4),
        "lora_bleu_char": round(lora_bleu.score, 4),
        "bleu_delta": round(lora_bleu.score - base_bleu.score, 4),
        "base_chrf": round(base_chrf.score, 4),
        "lora_chrf": round(lora_chrf.score, 4),
        "chrf_delta": round(lora_chrf.score - base_chrf.score, 4),
        "mean_sentence_chrf_delta": round(sum(deltas) / len(deltas), 4),
        "mean_sentence_chrf_delta_ci95": [
            round(bootstrap_means[25], 4), round(bootstrap_means[975], 4)
        ],
        "sentence_chrf_wins": wins,
        "sentence_chrf_ties": ties,
        "sentence_chrf_losses": losses,
        "changed_output_rate": round(sum(row["changed"] for row in rows) / len(rows), 4),
        "base_empty_rate": round(sum(not row["base_prediction"].strip() for row in rows) / len(rows), 4),
        "lora_empty_rate": round(sum(not row["lora_prediction"].strip() for row in rows) / len(rows), 4),
        "chrf_signature": "chrF(char_order=6, word_order=0)",
        "sacrebleu_version": sacrebleu.__version__,
    }
    return rows, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset_index", "source", "reference", "base_prediction", "lora_prediction",
        "base_bleu", "lora_bleu", "base_chrf", "lora_chrf", "chrf_delta",
        "reference_chars", "base_chars", "lora_chars", "changed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def latex_escape(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}")
                .replace("&", r"\&").replace("%", r"\%")
                .replace("#", r"\#").replace("_", r"\_")
                .replace("{", r"\{").replace("}", r"\}"))


def write_latex(summary: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path, max_examples: int) -> None:
    base_empty_pct = f"{summary['base_empty_rate'] * 100:.2f}\\%"
    lora_empty_pct = f"{summary['lora_empty_rate'] * 100:.2f}\\%"
    table = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Model & BLEU (char) & chrF & Empty rate \\", r"\midrule",
        f"Base Qwen3-0.6B & {summary['base_bleu_char']:.2f} & {summary['base_chrf']:.2f} & {base_empty_pct} \\",
        f"LoRA fine-tuned & {summary['lora_bleu_char']:.2f} & {summary['lora_chrf']:.2f} & {lora_empty_pct} \\",
        r"\bottomrule", r"\end{tabular}", "",
        rf"% Improvements: BLEU {summary['bleu_delta']:+.2f}, chrF {summary['chrf_delta']:+.2f}.",
    ]
    (output_dir / "metrics_table.tex").write_text("\n".join(table), encoding="utf-8")

    half = max(1, max_examples // 2)
    ranked = (sorted(rows, key=lambda row: row["chrf_delta"], reverse=True)[:half]
              + sorted(rows, key=lambda row: row["chrf_delta"])[:half])
    lines = [r"\begin{tabular}{p{0.17\linewidth}p{0.25\linewidth}p{0.25\linewidth}p{0.25\linewidth}}", r"\toprule", r"Source & Reference & Base & LoRA \\", r"\midrule"]
    for row in ranked:
        cells = [latex_escape(row[key])[:400] for key in ("source", "reference", "base_prediction", "lora_prediction")]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "examples_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    run_config_path = output_dir / "run_config.json"
    current_run_config = vars(args)
    reuse_existing = False
    saved_rows: dict[int, dict[str, Any]] = {}
    if run_config_path.exists() and prediction_path.exists():
        try:
            prior_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            if prior_config == current_run_config:
                with prediction_path.open(encoding="utf-8") as handle:
                    saved_rows = {
                        int(row["dataset_index"]): row
                        for line in handle if line.strip()
                        for row in [json.loads(line)]
                    }
                reuse_existing = bool(saved_rows)
        except (OSError, ValueError, KeyError, TypeError):
            saved_rows = {}
    run_config_path.write_text(json.dumps(current_run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Loading held-out split: {args.dataset_name}/{args.dataset_config}:{args.split}", flush=True)
    rows = select_examples(args)
    if reuse_existing:
        for row in rows:
            previous = saved_rows.get(row["dataset_index"])
            if (previous and previous.get("source") == row["source"]
                    and previous.get("reference") == row["reference"]):
                for key in ("base_prediction", "lora_prediction"):
                    if key in previous:
                        row[key] = previous[key]
    prediction_path.open("w", encoding="utf-8").close()
    reused_rows = [row for row in rows if "base_prediction" in row and "lora_prediction" in row]
    if reused_rows:
        with prediction_path.open("a", encoding="utf-8") as handle:
            for row in reused_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Evaluating {len(rows)} examples with batch size {args.batch_size}", flush=True)
    missing = [row for row in rows if "base_prediction" not in row or "lora_prediction" not in row]
    if missing:
        tokenizer, model = load_model(args.base_model, args.adapter)
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start : start + args.batch_size]
            sources = [row["source"] for row in batch]
            base_outputs = generate_batch(tokenizer, model, sources, args, False)
            lora_outputs = generate_batch(tokenizer, model, sources, args, True)
            for row, prediction in zip(batch, base_outputs):
                row["base_prediction"] = prediction
            for row, prediction in zip(batch, lora_outputs):
                row["lora_prediction"] = prediction
            with prediction_path.open("a", encoding="utf-8") as handle:
                for row in batch:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (start // args.batch_size + 1) % 10 == 0 or start + args.batch_size >= len(missing):
                completed = len(rows) - len(missing) + min(start + args.batch_size, len(missing))
                print(f"Generated {completed}/{len(rows)} examples", flush=True)
    else:
        print(f"Reusing {len(rows)} saved predictions; skipping model inference", flush=True)

    rows, summary = score_rows(rows)
    summary.update(
        {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "dataset": f"{args.dataset_name}/{args.dataset_config}",
            "split": args.split,
            "seed": args.seed,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    write_jsonl(prediction_path, rows)
    write_csv(output_dir / "sentence_metrics.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex(summary, rows, output_dir, args.max_examples_in_latex)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Raw predictions and report tables saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
