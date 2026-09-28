from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .config import TrainConfig
from .data import TranslationFormatter, load_translation_splits, tokenize_splits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 with LoRA for translation.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def _load_model(config: TrainConfig):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if config.use_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
    model_kwargs = {"torch_dtype": compute_dtype if torch.cuda.is_available() else torch.float32}
    if quantization_config is not None:
        model_kwargs.update({"quantization_config": quantization_config, "device_map": "auto"})
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def _training_arguments(config: TrainConfig, output_dir: Path) -> TrainingArguments:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config.use_4bit and torch.cuda.is_available() else "adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=False,
    )
    if "eval_strategy" not in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def train(config: TrainConfig) -> Path:
    set_seed(config.seed)
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = load_translation_splits(config)
    formatter = TranslationFormatter(
        tokenizer=tokenizer,
        source_language=config.source_language,
        target_language=config.target_language,
        source_field=config.source_field,
        target_field=config.target_field,
        max_length=config.max_length,
    )
    datasets = tokenize_splits(raw, formatter)
    model = _load_model(config)
    arguments = _training_arguments(config, output_dir)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    trainer_kwargs = dict(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Training finished. Adapter saved to {output_dir / 'final'}")
    return output_dir / "final"


def main() -> None:
    args = _parse_args()
    config = TrainConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_eval_samples is not None:
        config.max_eval_samples = args.max_eval_samples
    if args.no_4bit:
        config.use_4bit = False
    train(config)


if __name__ == "__main__":
    main()
