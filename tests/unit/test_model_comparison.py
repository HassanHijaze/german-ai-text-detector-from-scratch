"""Holm correction, McNemar tests, error by group and prediction loading in 09_model-comparison/scripts/."""

import numpy as np
import pandas as pd
import pytest


# 16
def test_holm_known_values(mc):
    # Sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> raised to 0.06 (monotone)
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(mc.analyze.holm(p), [0.03, 0.06, 0.06, 0.02])


# 17
def test_holm_never_lowers_p_values(mc):
    p = np.random.default_rng(0).uniform(0, 0.2, 25)
    adjusted = mc.analyze.holm(p)
    assert np.all(adjusted >= p)
    assert np.all(adjusted <= 1)


# 18
def test_mcnemar_identical_predictions_not_significant(mc):
    hits = np.random.default_rng(0).random(200) < 0.9
    correct = pd.DataFrame({"Model A": hits, "Model B": hits})
    row = mc.analyze.mcnemar_table(correct).iloc[0]
    assert row["a_right_b_wrong"] == row["a_wrong_b_right"] == 0
    assert not row["significant"]
    assert row["result"] == "no significant difference"


# 19
def test_mcnemar_finds_clearly_better_model(mc):
    model_a = np.ones(200, dtype=bool)                  # right on every text
    model_b = np.arange(200) % 4 != 0                   # wrong on every 4th text
    correct = pd.DataFrame({"Model A": model_a, "Model B": model_b})
    row = mc.analyze.mcnemar_table(correct).iloc[0]
    assert (row["a_right_b_wrong"], row["a_wrong_b_right"]) == (50, 0)
    assert row["significant"]
    assert row["result"] == "Model A better"


# 20
def test_error_by_group(mc):
    meta = pd.DataFrame({
        "label": [0, 0, 1, 1, 2, 2],
        "source": ["ssoar", "pedocs", "ssoar", "pedocs", "ssoar", "pedocs"],
        "generator_model": ["none", "none", "gemini", "deepseek-flash", "gemini", "deepseek-flash"],
    })
    correct = pd.DataFrame({
        "Model A": [True, False, True, False, False, True],
        "Model B": [True, True, True, True, False, False],
    })
    by_generator = mc.analyze.error_by_group(correct, meta["generator_model"], mask=meta["label"] != 0)
    assert set(by_generator.index) == {"gemini", "deepseek-flash"}
    assert by_generator.loc["gemini", "samples"] == 2
    assert by_generator.loc["gemini", "Model A"] == pytest.approx(0.5)
    assert by_generator.loc["deepseek-flash", "Model A"] == pytest.approx(0.5)
    assert by_generator.loc["gemini", "Model B"] == pytest.approx(0.5)
    assert by_generator.loc["deepseek-flash", "Model B"] == pytest.approx(0.5)

    by_source = mc.analyze.error_by_group(correct, meta["source"])
    assert by_source.loc["ssoar", "samples"] == 3
    assert by_source.loc["ssoar", "Model A"] == pytest.approx(1 / 3)
    assert by_source.loc["pedocs", "Model A"] == pytest.approx(2 / 3)
    assert by_source.loc["ssoar", "Model B"] == pytest.approx(1 / 3)
    assert by_source.loc["pedocs", "Model B"] == pytest.approx(1 / 3)


# 21
def write_predictions(path, ids, labels):
    n = len(ids)
    pd.DataFrame({
        "id": ids, "label": labels, "word_count": [100] * n, "source": ["ssoar"] * n,
        "generator_model": ["gemini"] * n, "pred": labels,
        "p_human": [0.8] * n, "p_ai_generated": [0.1] * n, "p_ai_assisted": [0.1] * n,
    }).to_csv(path, index=False)


@pytest.fixture
def two_model_setup(mc, tmp_path, monkeypatch):
    """Point load_predictions at two small prediction files in a temporary folder."""
    monkeypatch.setattr(mc.common, "PRED_DIR", tmp_path)
    monkeypatch.setattr(mc.common, "MODELS", {"model-a": "Model A", "model-b": "Model B"})
    monkeypatch.setattr(mc.common, "N_TEST", 4)
    return tmp_path


def test_matching_prediction_files_load(mc, two_model_setup):
    write_predictions(two_model_setup / "model-a-test-predictions.csv", [1, 2, 3, 4], [0, 1, 2, 0])
    write_predictions(two_model_setup / "model-b-test-predictions.csv", [1, 2, 3, 4], [0, 1, 2, 0])
    meta, pred, correct, conf = mc.common.load_predictions()
    assert list(pred.columns) == ["Model A", "Model B"]
    assert correct.all().all()


@pytest.mark.parametrize("ids_b, labels_b, message", [
    ([1, 2, 4, 3], [0, 1, 2, 0], "ids are not aligned"),
    ([1, 2, 3, 4], [0, 1, 1, 0], "labels are not aligned"),
])
def test_mismatched_prediction_files_raise(mc, two_model_setup, ids_b, labels_b, message):
    write_predictions(two_model_setup / "model-a-test-predictions.csv", [1, 2, 3, 4], [0, 1, 2, 0])
    write_predictions(two_model_setup / "model-b-test-predictions.csv", ids_b, labels_b)
    with pytest.raises(ValueError, match=message):
        mc.common.load_predictions()
