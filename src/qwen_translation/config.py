from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DriveConfig:
    enabled: bool = True
    mount_point: str = "/content/drive"
    subdirectory: str = "MyDrive/qwen3-translation"


@dataclass
class TrainConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    dataset_name: str = "Helsinki-NLP/opus-100"
    dataset_config: str = "en-zh"
    source_language: str = "English"
    target_language: str = "Chinese"
    source_field: str = "en"
    target_field: str = "zh"
    max_train_samples: int = 20_000
    max_eval_samples: int = 1_000
    validation_size: float = 0.05
    seed: int = 42
    max_length: int = 768
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 2
    output_dir: str = "checkpoints/qwen3-0.6b-en-zh-lora"
    dataset_cache_dir: str | None = None
    drive: DriveConfig = field(default_factory=DriveConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        drive = DriveConfig(**(raw.pop("drive", {}) or {}))
        known = {key: value for key, value in raw.items() if key in cls.__dataclass_fields__}
        return cls(**known, drive=drive)

    def resolved_output_dir(self) -> Path:
        """Return a Drive path when the launcher has mounted Drive."""
        output = Path(self.output_dir).expanduser()
        if output.is_absolute():
            return output
        mount = Path(self.drive.mount_point)
        if self.drive.enabled and mount.exists():
            return mount / self.drive.subdirectory / output
        return output

    def resolved_cache_dir(self) -> str | None:
        if not self.dataset_cache_dir:
            return None
        cache = Path(self.dataset_cache_dir).expanduser()
        if cache.is_absolute():
            return str(cache)
        return str(self.resolved_output_dir().parent / cache)

