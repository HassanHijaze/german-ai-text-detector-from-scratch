"""Shared helpers for the fine-tuning scripts: config, paths, dataset and model names."""

from pathlib import Path

import yaml

# ---------- Paths ----------
SCRIPTS_DIR = Path(__file__).resolve().parent     # <repo>/07_fine-tuning/scripts
FT_DIR = SCRIPTS_DIR.parent                       # <repo>/07_fine-tuning
REPO_ROOT = FT_DIR.parent                         # <repo>
CONFIG_PATH = FT_DIR / "config.yaml"
ARTIFACTS_DIR = FT_DIR / "artifacts"              # git-ignored
CHECKPOINTS_DIR = FT_DIR / "checkpoints"          # git-ignored
RESULTS_DIR = FT_DIR / "results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("dataset_path", "labels", "training", "models"):
        if key not in config:
            raise KeyError(f"config.yaml is missing the '{key}' section")
    config["labels"] = {int(k): v for k, v in config["labels"].items()}
    return config


CONFIG = load_config()
MODEL_NAMES = list(CONFIG["models"])


class UnknownModelError(ValueError):
    pass


def validate_model_name(name: str) -> str:
    if name not in CONFIG["models"]:
        raise UnknownModelError(
            f"Unknown model '{name}'. Valid names: {', '.join(MODEL_NAMES)}"
        )
    return name


def model_spec(name: str) -> dict:
    return CONFIG["models"][validate_model_name(name)]


def dataset_path() -> Path:
    return REPO_ROOT / CONFIG["dataset_path"]


def artifact_dir(name: str) -> Path:
    return ARTIFACTS_DIR / f"{validate_model_name(name)}-3class"


def checkpoint_dir(name: str) -> Path:
    return CHECKPOINTS_DIR / validate_model_name(name)


def results_dir(name: str) -> Path:
    return RESULTS_DIR / validate_model_name(name)


def new_run_dir(name: str) -> Path:
    """Where new evaluation runs write, so the reported results in results/<name>/ are never overwritten."""
    return results_dir(name) / "new-run"


def id2label() -> dict:
    """Label names stored in the model config (HUMAN / AI / AI_ASSISTED, as in the notebooks)."""
    return {i: v["model_label"] for i, v in CONFIG["labels"].items()}


def label2id() -> dict:
    return {v: k for k, v in id2label().items()}


def load_dataset():
    """Load the saved three-class dataset (train / validation / test)."""
    from datasets import load_from_disk

    path = dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. The dataset is private: place hf-dataset-3class-full "
            f"there (see 04_hf-dataset) or change dataset_path in {CONFIG_PATH.name}."
        )
    dataset = load_from_disk(str(path))
    for split in ("train", "validation", "test"):
        if split not in dataset:
            raise KeyError(f"Dataset at {path} has no '{split}' split")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """Tokenize every split exactly like the notebooks (truncate to max_length, keep only label)."""
    max_length = CONFIG["training"]["max_length"]

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    columns_to_remove = [c for c in dataset["train"].column_names if c != "label"]
    return dataset.map(tokenize_batch, batched=True, remove_columns=columns_to_remove,
                       desc="Tokenizing dataset")


def data_collator(tokenizer):
    import torch
    from transformers import DataCollatorWithPadding

    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )


def precision_flags() -> dict:
    """bf16 if the GPU supports it, else fp16 on GPU, else full precision (as in the notebooks)."""
    import torch

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": use_bf16, "fp16": torch.cuda.is_available() and not use_bf16}
