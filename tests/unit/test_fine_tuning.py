"""Config, model names, length bins and calibration in 07_fine-tuning/scripts/."""

import warnings

import numpy as np
import pytest

MODEL_NAMES = ["mmbert-small", "distilbert-german", "mmbert-base", "xlm-roberta-base", "gelectra-large"]


# 10
def test_config_has_five_models_and_training_settings(ft):
    config = ft.common.CONFIG
    assert ft.common.MODEL_NAMES == MODEL_NAMES
    for spec in config["models"].values():
        assert {"display_name", "base_model", "hf_repo"} <= set(spec)
    training = config["training"]
    assert training["learning_rate"] == pytest.approx(2e-5)
    assert training["per_device_train_batch_size"] == 8
    assert training["per_device_eval_batch_size"] == 16
    assert training["num_train_epochs"] == 3
    assert training["weight_decay"] == pytest.approx(0.01)
    assert training["warmup_steps"] == 235
    assert training["early_stopping_patience"] == 1
    assert training["metric_for_best_model"] == "accuracy"
    assert training["seed"] == 17
    assert training["max_length"] == 512
    assert training["save_total_limit"] == 2


# 11
def test_unknown_model_name_lists_valid_names(ft):
    with pytest.raises(ft.common.UnknownModelError) as error:
        ft.common.validate_model_name("bert-large")
    for name in MODEL_NAMES:
        assert name in str(error.value)


# 12
@pytest.mark.parametrize("words, expected_bin", [(70, "≤70"), (71, "71–150"), (750, "501–750"), (751, "751+")])
def test_length_bins(ft, words, expected_bin):
    text = " ".join(["Wort"] * words)
    y_true = np.array([0])
    probabilities = np.array([[0.8, 0.1, 0.1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mean of the empty AI classes in a one-text bin
        table = ft.evaluate.length_bias("test", [text], y_true, probabilities)
    assert list(table["length_bin"]) == [expected_bin]


def overconfident_logits(n=2000, accuracy=0.7, scale=8.0, seed=0):
    """Logits that put ~all probability on one class, but are right only 70% of the time."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    predicted = np.where(rng.random(n) < accuracy, labels, (labels + rng.integers(1, 3, n)) % 3)
    logits = rng.normal(0, 0.5, (n, 3))
    logits[np.arange(n), predicted] += scale
    return logits, labels


# 13
def test_temperature_softens_overconfident_logits(ft):
    logits, labels = overconfident_logits()
    temperature = ft.evaluate.fit_temperature(logits, labels)
    assert temperature > 1
    before = ft.evaluate.probabilities_from_logits(logits).max(axis=1).mean()
    after = ft.evaluate.probabilities_from_logits(logits, temperature).max(axis=1).mean()
    assert after < before


# 14
def test_calibration_keeps_the_predicted_class(ft):
    logits, labels = overconfident_logits(seed=1)
    temperature = ft.evaluate.fit_temperature(logits, labels)
    uncalibrated = ft.evaluate.probabilities_from_logits(logits).argmax(axis=1)
    calibrated = ft.evaluate.probabilities_from_logits(logits, temperature).argmax(axis=1)
    np.testing.assert_array_equal(calibrated, uncalibrated)


# 15
def test_label_mapping(ft):
    labels = ft.common.CONFIG["labels"]
    assert {i: labels[i]["name"] for i in labels} == {0: "human", 1: "ai_generated", 2: "ai_assisted"}
    assert ft.common.id2label() == {0: "HUMAN", 1: "AI", 2: "AI_ASSISTED"}
    assert ft.common.label2id() == {"HUMAN": 0, "AI": 1, "AI_ASSISTED": 2}
