"""Training-serving parity: the evaluation pipeline (07_fine-tuning) and the web app
(11_deployment) must give the same probabilities for the same texts.

Uses the fine-tuned mmBERT-base (from DETECTOR_MODEL or Hugging Face) with the temperature from
its own detector-config.json. Marked behavioral: runs with pytest -m behavioral, not in a plain pytest.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.behavioral

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "11_deployment" / "app.py"
DEMO_TEXTS = REPO_ROOT / "tests" / "behavioral" / "texts" / "demo_texts.json"

SHORTER_CUT = 256   # the long text cut to this many tokens must give a different result than at 512
ATOL = 1e-5

SHORT = [
    "Die Studie zeigt deutliche Unterschiede zwischen den Schulen.",
    "Kinder lernen am besten durch eigenes Ausprobieren.",
]
# 50-100 words: accepted by detect() and analysed in a single window
MEDIUM = (
    "Im Mittelpunkt dieses Beitrags steht die Frage, wie Lehrkräfte an Grundschulen mit der zunehmenden "
    "Heterogenität ihrer Klassen umgehen. Grundlage sind zwanzig Leitfadeninterviews, die zwischen 2017 und "
    "2019 in drei Bundesländern geführt wurden. Die Auswertung zeigt, dass die Befragten vor allem fehlende "
    "Zeit und unzureichende Fortbildungsangebote als Hindernisse nennen. Zugleich betonen viele von ihnen, "
    "dass sich durch kooperative Lernformen neue Spielräume für individuelle Förderung ergeben."
)
CITATIONS = (
    "Nach Esser (2000: 285f.) hängt die „Sozialintegration“ von Migrantinnen und Migranten von vier Ebenen ab. "
    "Müller und Schmidt (2015) ergänzen, dass rund 37 % der Befragten über Diskriminierung berichten."
)
# Over 512 tokens (about 900 with the mmBERT tokenizer)
LONG = " ".join(["Die Ergebnisse der Befragung werden im folgenden Abschnitt ausführlich diskutiert."] * 60)


def demo_texts() -> list[str]:
    texts = json.loads(DEMO_TEXTS.read_text(encoding="utf-8"))["texts"]
    return [" ".join(texts[label]) for label in ("human", "ai_assisted", "ai_generated")]


# ---------- Loading both pipelines ----------
def load_app(model_dir: Path) -> types.ModuleType:
    """11_deployment/app.py with the real model from model_dir (MODEL_DIR) and a stubbed Gradio."""
    stub_gradio = MagicMock(name="gradio")
    stub_gradio.Error = type("StubGradioError", (Exception,), {})
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_DIR", str(model_dir))
        mp.setitem(sys.modules, "gradio", stub_gradio)
        spec = importlib.util.spec_from_file_location(f"deployment_app_parity_{model_dir.name}", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def real_model_dir(source: str) -> Path:
    """A local folder for the model: DETECTOR_MODEL as it is, or a snapshot from Hugging Face."""
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(source))


@pytest.fixture(scope="module")
def parity(ft, detector_model_source, tmp_path_factory):
    """Runs both pipelines once and returns everything the tests compare."""
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = real_model_dir(detector_model_source)
    detector_config = json.loads((model_dir / "detector-config.json").read_text(encoding="utf-8"))
    temperature = float(detector_config["temperature"])
    texts = SHORT + [MEDIUM, CITATIONS, " ".join([MEDIUM, CITATIONS, MEDIUM])] + demo_texts() + [LONG]

    with pytest.MonkeyPatch.context() as mp:
        # Same device (CPU) for both pipelines; on the CPU the Trainer runs in fp32 (no bf16 / fp16)
        mp.setenv("CUDA_VISIBLE_DEVICES", "")
        mp.setattr(torch.cuda, "is_available", lambda: False)

        # Serving
        app = load_app(model_dir)
        app_probs = np.asarray(app.predict(texts))
        with pytest.MonkeyPatch.context() as no_calibration:
            no_calibration.setattr(app, "TEMPERATURE", 1.0)
            app_raw = np.asarray(app.predict(texts))
        detect_plain = app.detect(MEDIUM)[0]
        detect_newline = app.detect(MEDIUM + "\n")[0]

        # Evaluation (model loaded like evaluate() loads it)
        eval_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eval_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        work_dir = tmp_path_factory.mktemp("parity-trainer")
        predict = ft.evaluate.predict_probabilities
        eval_probs = predict(eval_model, eval_tokenizer, texts, temperature, work_dir)
        eval_raw = predict(eval_model, eval_tokenizer, texts, 1.0, work_dir)
        eval_newline = predict(eval_model, eval_tokenizer, [MEDIUM + "\n"], temperature, work_dir)[0]
        eval_ids = ft.common.tokenize_dataset(DatasetDict({"train": Dataset.from_dict({"text": texts})}),
                                              eval_tokenizer)["train"]["input_ids"]

        # The long text cut by hand to its first n tokens (the final special token is kept),
        # fed to the model directly, without the tokenizer's truncation
        full_ids = eval_tokenizer(LONG)["input_ids"]
        long_cuts = {}
        for n in (512, SHORTER_CUT):
            ids = full_ids[:n - 1] + full_ids[-1:]
            with torch.inference_mode():
                logits = eval_model(input_ids=torch.tensor([ids])).logits.numpy()
            long_cuts[n] = (ids, ft.evaluate.probabilities_from_logits(logits, temperature)[0])

    return types.SimpleNamespace(
        version="mmbert", model_dir=model_dir, detector_config=detector_config, temperature=temperature,
        texts=texts, app=app, app_probs=app_probs, app_raw=app_raw, detect_plain=detect_plain,
        detect_newline=detect_newline, eval_model=eval_model, eval_tokenizer=eval_tokenizer,
        eval_probs=eval_probs, eval_raw=eval_raw, eval_newline=eval_newline, eval_ids=eval_ids,
        full_ids=full_ids, long_cuts=long_cuts, ft=ft,
    )


# ---------- The checks ----------
def test_both_pipelines_ran_on_cpu_in_fp32(parity):
    for name, model in [("app", parity.app.model), ("evaluation", parity.eval_model)]:
        parameter = next(model.parameters())
        assert parameter.device.type == "cpu", f"{name} model is on {parameter.device}"
        assert parameter.dtype == torch.float32, f"{name} model is in {parameter.dtype}"


def test_texts_have_no_surrounding_whitespace_and_one_is_too_long(parity):
    assert all(t == t.strip() for t in parity.texts)
    tokens = [len(parity.eval_tokenizer(t)["input_ids"]) for t in parity.texts]
    words = [len(t.split()) for t in parity.texts]
    assert max(tokens) > 512 and min(words) < 10, (tokens, words)   # short, medium and over 512 tokens
    assert 50 <= parity.app.real_word_count(MEDIUM) and len(MEDIUM.split()) <= 100   # one window in detect()


def test_results_differ_between_texts(parity):
    """The model must react to its input, otherwise matching probabilities would prove nothing:
    results must differ between texts by far more (100x) than the tolerance."""
    spread = parity.eval_probs.max(axis=0) - parity.eval_probs.min(axis=0)
    print(f"[{parity.version}] spread of the probabilities across texts: {np.round(spread, 4)}")
    assert spread.max() > 100 * ATOL


def test_same_tokenization(parity):
    app_ids = parity.app.tokenizer(parity.texts, truncation=True, max_length=parity.app.MAX_LENGTH)["input_ids"]
    for text, a, e in zip(parity.texts, app_ids, parity.eval_ids):
        assert a == e, f"token ids differ for {text[:40]!r}: app {len(a)} tokens, evaluation {len(e)} tokens"


def test_probabilities_match(parity):
    diff = np.abs(parity.app_probs - parity.eval_probs)
    print(f"[{parity.version}] largest probability difference: {diff.max():.3e}")
    for text, a, e in zip(parity.texts, parity.app_probs, parity.eval_probs):
        print(f"[{parity.version}] app {np.round(a, 6)}  evaluation {np.round(e, 6)}  {text[:40]!r}")
    assert np.allclose(parity.app_probs, parity.eval_probs, atol=ATOL), (
        f"largest difference {diff.max():.3e} (text {diff.max(axis=1).argmax()})"
    )


def test_same_predicted_class(parity):
    assert list(parity.app_probs.argmax(axis=1)) == list(parity.eval_probs.argmax(axis=1))


def test_same_calibration(parity):
    # Both use the temperature stored in detector-config.json
    assert parity.app.TEMPERATURE == parity.temperature
    # Same model output before calibration ...
    assert np.allclose(parity.app_raw, parity.eval_raw, atol=ATOL), (
        f"uncalibrated probabilities differ by up to {np.abs(parity.app_raw - parity.eval_raw).max():.3e}"
    )
    # ... the temperature changes the result ...
    assert not np.allclose(parity.app_probs, parity.app_raw, atol=ATOL)
    # ... and both apply it as softmax(logits / T) (log-probabilities differ from logits only by a constant)
    log_raw = np.log(parity.eval_raw)
    expected = parity.ft.evaluate.probabilities_from_logits(log_raw, parity.temperature)
    assert np.allclose(parity.app_probs, expected, atol=ATOL)
    assert np.allclose(parity.eval_probs, expected, atol=ATOL)


def test_same_truncation(parity):
    max_length = parity.ft.common.CONFIG["training"]["max_length"]
    assert parity.app.MAX_LENGTH == max_length == 512
    i = parity.texts.index(LONG)
    assert len(parity.full_ids) > 512
    assert parity.full_ids[-1] in parity.eval_tokenizer.all_special_ids
    assert np.allclose(parity.app_probs[i], parity.eval_probs[i], atol=ATOL)

    # Both pipelines keep exactly the first 512 tokens: the same ids as the cut by hand ...
    ids_512, probs_512 = parity.long_cuts[512]
    _, probs_shorter = parity.long_cuts[SHORTER_CUT]
    assert parity.eval_ids[i] == ids_512
    # ... and the same result as feeding those 512 tokens to the model directly
    assert np.allclose(parity.app_probs[i], probs_512, atol=ATOL)
    assert np.allclose(parity.eval_probs[i], probs_512, atol=ATOL)
    # The tokens near the cut matter, so the equality above is not trivially true
    print(f"[{parity.version}] long text ({len(parity.full_ids)} tokens): pipelines {np.round(parity.eval_probs[i], 6)}, "
          f"first 512 tokens {np.round(probs_512, 6)}, first {SHORTER_CUT} tokens {np.round(probs_shorter, 6)}")
    assert not np.allclose(probs_512, probs_shorter, atol=ATOL), (
        f"cutting at {SHORTER_CUT} or 512 tokens gives the same result, so the cut position is not tested"
    )


def test_same_label_order(parity):
    labels = parity.ft.common.CONFIG["labels"]
    assert [labels[i]["name"] for i in range(3)] == ["human", "ai_generated", "ai_assisted"]
    assert parity.app.LABELS == [labels[i]["display"] for i in range(3)] == ["Human", "AI-generated", "AI-assisted"]
    expected_id2label = {0: "HUMAN", 1: "AI", 2: "AI_ASSISTED"}
    assert parity.app.model.config.id2label == expected_id2label
    assert parity.eval_model.config.id2label == expected_id2label
    assert parity.detector_config["label_mapping"] == {"human": 0, "ai": 1, "ai_assisted": 2}


def test_app_ignores_surrounding_whitespace(parity):
    """Known, intended difference: detect() strips the input, the evaluation pipeline does not.
    A trailing newline therefore leaves the app's result unchanged (the evaluation result may change)."""
    assert parity.detect_newline == parity.detect_plain
    i = parity.texts.index(MEDIUM)
    app_overall = [parity.detect_plain[label] for label in parity.app.LABELS]
    assert np.allclose(app_overall, parity.eval_probs[i], atol=ATOL)
    print(f"[{parity.version}] evaluation without / with trailing newline: "
          f"{np.round(parity.eval_probs[i], 6)} / {np.round(parity.eval_newline, 6)}")
