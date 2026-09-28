from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TranslationFormatter:
    tokenizer: PreTrainedTokenizerBase
    source_language: str
    target_language: str
    source_field: str
    target_field: str
    max_length: int

    def _chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            # Older Transformers versions do not expose enable_thinking.
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, example: dict[str, Any]) -> dict[str, list[int]]:
        source = str(example[self.source_field]).strip()
        target = str(example[self.target_field]).strip()
        system = (
            "You are a professional translator. Translate the user's text from "
            f"{self.source_language} to {self.target_language}. Return only the translation."
        )
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": source},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": target}]
        prompt_ids = self._chat_template(prompt_messages, add_generation_prompt=True)
        full_ids = self._chat_template(full_messages, add_generation_prompt=False)

        # Chat templates should make the prompt a prefix of the supervised
        # sequence. The fallback avoids training on the prompt if a template
        # inserts a different assistant marker.
        prefix_len = len(prompt_ids)
        if full_ids[:prefix_len] != prompt_ids:
            prefix_len = min(prefix_len, len(full_ids))
        input_ids = full_ids[: self.max_length]
        labels = input_ids.copy()
        for index in range(min(prefix_len, len(labels))):
            labels[index] = -100
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _select(dataset: Dataset, limit: int, seed: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def load_translation_splits(config: Any) -> DatasetDict:
    raw = load_dataset(
        config.dataset_name,
        config.dataset_config,
        cache_dir=config.resolved_cache_dir(),
    )
    train = raw["train"]
    if "validation" in raw:
        evaluation = raw["validation"]
    elif "test" in raw:
        evaluation = raw["test"]
    else:
        split = train.train_test_split(test_size=config.validation_size, seed=config.seed)
        train, evaluation = split["train"], split["test"]
    return DatasetDict(
        train=_select(train, config.max_train_samples, config.seed),
        validation=_select(evaluation, config.max_eval_samples, config.seed),
    )


def tokenize_splits(raw: DatasetDict, formatter: TranslationFormatter) -> DatasetDict:
    columns = raw["train"].column_names
    return raw.map(formatter, remove_columns=columns, desc="Formatting translation examples")

