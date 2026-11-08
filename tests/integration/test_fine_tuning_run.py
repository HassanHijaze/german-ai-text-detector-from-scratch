"""Train and evaluate with 07_fine-tuning/scripts on a tiny fake dataset and a tiny random model."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"  # ~87k parameters, no classification head
NAME = "tiny-test"
WORDS = ["Bildung", "Schule", "Studie", "Daten", "Ergebnis", "Analyse", "Gesellschaft", "Forschung",
         "Methode", "Lernen", "Kinder", "Lehrkräfte", "Befragung", "Wandel", "Politik", "Theorie"]


def fake_split(n_per_class, rng, start_id):
    rows = []
    for label in range(3):
        for i in range(n_per_class):
            length = int(rng.choice([40, 90, 200, 400, 600, 800]))  # covers several length bins
            text = " ".join(rng.choice(WORDS, length)) + "."
            rows.append({
                "id": str(start_id + len(rows)), "text": text, "label": label,
                "source": "ssoar", "generator_model": None if label == 0 else "gemini",
            })
    return rows


@pytest.fixture
def tiny_dataset(tmp_path):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

    features = Features({"id": Value("string"), "text": Value("string"),
                         "label": ClassLabel(names=["human", "ai_generated", "ai_assisted"]),
                         "source": Value("string"), "generator_model": Value("string")})
    rng = np.random.default_rng(0)
    splits = {"train": fake_split(10, rng, 0), "validation": fake_split(3, rng, 100), "test": fake_split(3, rng, 200)}
    path = tmp_path / "hf-dataset-3class-full"
    DatasetDict({s: Dataset.from_list(rows, features=features) for s, rows in splits.items()}).save_to_disk(str(path))
    return path


@pytest.mark.slow
def test_train_and_evaluate_tiny_model(ft_full, tiny_dataset, tmp_path, monkeypatch):
    common = ft_full.common

    # CPU only, even if a GPU is present
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Override the config in memory (config.yaml is not touched)
    monkeypatch.setitem(common.CONFIG, "dataset_path", str(tiny_dataset))
    monkeypatch.setitem(common.CONFIG["models"], NAME,
                        {"display_name": "Tiny test model", "base_model": TINY_MODEL, "hf_repo": TINY_MODEL})
    for key, value in {"num_train_epochs": 1, "per_device_train_batch_size": 4, "per_device_eval_batch_size": 8,
                       "warmup_steps": 0, "max_length": 64, "logging_steps": 5}.items():
        monkeypatch.setitem(common.CONFIG["training"], key, value)
    monkeypatch.setattr(common, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(common, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path / "results")

    ft_full.train.train(NAME)
    ft_full.evaluate.evaluate(NAME, from_hub=False)

    model_dir = tmp_path / "artifacts" / f"{NAME}-3class"
    assert (model_dir / "config.json").exists()
    assert any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").exists()
    assert (model_dir / "tokenizer_config.json").exists()

    detector_config = json.loads((model_dir / "detector-config.json").read_text(encoding="utf-8"))
    assert detector_config["temperature"] > 0
    assert detector_config["label_mapping"] == {"human": 0, "ai": 1, "ai_assisted": 2}

    out = tmp_path / "results" / NAME / "new-run"
    for file in [f"{NAME}-metrics.csv", f"{NAME}-3-class-confusion-matrix.csv", f"{NAME}-confmat-3class.png",
                 f"{NAME}-confmat-human-ai.csv", f"{NAME}-confmat-human-ai.png", f"{NAME}-length-bias-3class.csv"]:
        assert (out / file).exists(), f"{file} was not created"
        assert (out / file).stat().st_size > 0

    metrics = pd.read_csv(out / f"{NAME}-metrics.csv")
    assert list(metrics["split"]) == ["train", "validation", "test"]
    assert pd.read_csv(out / f"{NAME}-3-class-confusion-matrix.csv", index_col=0).values.sum() == 9
