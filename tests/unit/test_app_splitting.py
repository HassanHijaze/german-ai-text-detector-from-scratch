"""Sentence splitting, windows and rendering in 11_deployment/app.py."""

import html
import re

import pytest


def sentences(app, text):
    return [text[s:e] for s, e in app.sentence_spans(text)]


def strip_tags(rendered):
    return html.unescape(re.sub(r"<[^>]*>", "", rendered))


SCORES = {"Human": 0.6, "AI-generated": 0.1, "AI-assisted": 0.3}


# 1
def test_simple_sentences_are_split(app):
    assert sentences(app, "Satz eins. Satz zwei! Satz drei?") == ["Satz eins.", "Satz zwei!", "Satz drei?"]


# 2
@pytest.mark.parametrize("text", [
    "Das gilt z.B. auch hier.",
    "Dazu vgl. Kapitel zwei.",
    "Er bzw. sie kommt morgen.",
    "Heute spricht Dr. Müller.",
])
def test_abbreviations_do_not_end_a_sentence(app, text):
    assert sentences(app, text) == [text]


# 3
def test_ordinal_numbers_do_not_end_a_sentence(app):
    text = "Wir treffen uns am 3. Mai in Berlin."
    assert sentences(app, text) == [text]


# 4
@pytest.mark.parametrize("text, expected", [
    ("Erster Absatz ohne Punkt\n\nZweiter Absatz", ["Erster Absatz ohne Punkt", "Zweiter Absatz"]),
    ("Das gilt z.B.\n\nNeuer Absatz.", ["Das gilt z.B.", "Neuer Absatz."]),
    ("Satz eins.\n   \nSatz zwei.", ["Satz eins.", "Satz zwei."]),
])
def test_empty_line_starts_a_new_sentence(app, text, expected):
    assert sentences(app, text) == expected


# 5
def test_text_under_100_words_gives_one_window(app):
    text = " ".join(f"Das ist Satz Nummer {i} hier." for i in range(15))  # 90 words
    words = [app.word_count(s) for s in sentences(app, text)]
    assert sum(words) < 100
    assert app.make_windows(words) == [(0, len(words))]


def long_text_words():
    """Sentence lengths of a long text: 60 sentences of 7 to 18 words (about 750 words)."""
    return [7 + (i * 5) % 12 for i in range(60)]


# 6
def test_every_sentence_is_in_a_window(app):
    words = long_text_words()
    windows = app.make_windows(words)
    covered = {i for a, b in windows for i in range(a, b)}
    assert covered == set(range(len(words)))


# 7
def test_neighbouring_windows_overlap(app):
    windows = app.make_windows(long_text_words())
    assert len(windows) > 1
    for (a1, b1), (a2, b2) in zip(windows, windows[1:]):
        assert a2 < b1, f"windows {(a1, b1)} and {(a2, b2)} do not overlap"


# 8
def test_rendered_text_equals_input(app):
    text = ("Erster Absatz mit z.B. einer Abkürzung. Zweiter Satz & mehr <Text>!\n\n"
            "  Zweiter Absatz \"in Anführungszeichen\" am 3. Mai.\nNeue Zeile ohne Punkt")
    spans = [(s, e, SCORES) for s, e in app.sentence_spans(text)]
    assert strip_tags(app.render(text, spans)) == text


# 9
def test_html_in_input_is_escaped(app):
    text = "Normaler Satz. <script>alert('x')</script> Noch ein Satz."
    spans = [(s, e, SCORES) for s, e in app.sentence_spans(text)]
    rendered = app.render(text, spans)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert strip_tags(rendered) == text
