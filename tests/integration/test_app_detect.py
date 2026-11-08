"""detect() in 11_deployment/app.py, end to end with a stub model."""

import re

import pytest
import torch

MARKER = "Maschinentext"
HUMAN_BG, AI_GENERATED_BG = "#E3F0FF", "#FDE4E4"


class MarkerTokenizer:
    """Passes the texts through, so the stub model can look at them."""

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):                      # length check in detect()
            return {"input_ids": texts.split()}
        return {"texts": list(texts)}


class MarkerModel:
    """Predicts AI-generated for windows that contain the marker word, Human otherwise."""

    def __call__(self, texts, **kwargs):
        logits = [[0.0, 6.0, 0.0] if MARKER in t else [6.0, 0.0, 0.0] for t in texts]
        return type("Output", (), {"logits": torch.tensor(logits)})()


@pytest.fixture
def marker_app(app, monkeypatch):
    monkeypatch.setattr(app, "tokenizer", MarkerTokenizer())
    monkeypatch.setattr(app, "model", MarkerModel())
    monkeypatch.setattr(app, "TEMPERATURE", 1.0)
    return app


def span_labels(body):
    """Class of each coloured part in the rendered text, in order."""
    text_part = body.split('white-space:pre-wrap;">', 1)[1]
    return re.findall(r'<span title="([^:"]+):', text_part)


# 23
def test_human_then_ai_parts_are_coloured_in_order(marker_app):
    human = " ".join(f"Die Studie beschreibt im Abschnitt {i} die Lage der Schulen sehr genau." for i in range(15))
    ai = " ".join(f"Der {MARKER} fasst im Punkt {i} die wichtigsten Ergebnisse kurz zusammen." for i in range(15))
    assert len(human.split()) > 100 and len(ai.split()) > 100

    overall, body, notes = marker_app.detect(human + " " + ai)

    assert span_labels(body) == ["Human", "AI-generated"]
    text_part = body.split('white-space:pre-wrap;">', 1)[1]
    assert text_part.index(f"background:{HUMAN_BG}") < text_part.index(f"background:{AI_GENERATED_BG}")
    assert "display:flex; height:14px" in body                     # summary bar
    assert set(overall) == {"Human", "AI-generated", "AI-assisted"}
    assert sum(overall.values()) == pytest.approx(1.0)


# 24
def test_short_text_shows_warning(app):
    overall, body, notes = app.detect("Das ist ein kurzer Text mit nur wenigen Wörtern.")
    assert "Short text" in notes
    assert sum(overall.values()) == pytest.approx(1.0)


# 25
@pytest.mark.parametrize("text", ["", "   \n  ", None])
def test_empty_input_raises_clear_error(app, text):
    with pytest.raises(app.gr.Error, match="Please enter a German text"):
        app.detect(text)
