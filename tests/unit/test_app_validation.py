"""Input checks in 11_deployment/app.py: minimum and maximum length, language warning (stub model)."""

import pytest

GERMAN = (
    "Die vorliegende Untersuchung analysiert die Auswirkungen digitaler Lernplattformen auf die Motivation "
    "von Schülerinnen und Schülern der Sekundarstufe. Auf Grundlage einer Befragung an zwölf Schulen zeigt "
    "sich, dass vor allem die unmittelbare Rückmeldung und die freie Zeiteinteilung positiv bewertet werden, "
    "während fehlender persönlicher Kontakt zu Lehrkräften als Nachteil wahrgenommen wird. Die Ergebnisse "
    "deuten darauf hin, dass digitale Angebote den Präsenzunterricht ergänzen, aber nicht vollständig "
    "ersetzen sollten."
)

ENGLISH = (
    "This study examines how digital learning platforms affect the motivation of secondary school students. "
    "Based on a survey conducted at twelve schools, the results show that immediate feedback and flexible "
    "scheduling are rated positively, while the lack of personal contact with teachers is seen as a "
    "disadvantage. The findings suggest that digital tools should complement classroom teaching rather than "
    "replace it entirely."
)


def words(n: int) -> str:
    """A German text of exactly n words, all of them real words (with letters)."""
    base = GERMAN.split()
    return " ".join(base[i % len(base)] for i in range(n))


def test_german_text_gives_no_language_warning(app):
    assert app.real_word_count(GERMAN) >= app.MIN_INPUT_WORDS
    overall, body, notes = app.detect(GERMAN)
    assert "doesn't look German" not in notes


def test_english_text_gives_language_warning(app):
    assert app.real_word_count(ENGLISH) >= app.MIN_INPUT_WORDS
    overall, body, notes = app.detect(ENGLISH)
    assert "doesn't look German (detected: English)" in notes
    assert sum(overall.values()) == pytest.approx(1.0)  # the prediction still runs


def test_text_at_the_word_limit_is_accepted(app):
    text = words(app.MAX_WORDS)
    assert len(text.split()) == 3000
    overall, body, notes = app.detect(text)
    assert sum(overall.values()) == pytest.approx(1.0)


def test_text_over_the_word_limit_raises_error(app):
    with pytest.raises(app.gr.Error, match=r"The text has 3,001 words\. Please enter at most 3,000 words\."):
        app.detect(words(app.MAX_WORDS + 1))


@pytest.mark.parametrize("text, n_real", [
    ("hi", 1),
    ("1", 0),
    (" ".join(str(i) for i in range(1, 61)), 0),          # 60 numbers, no real words
])
def test_too_few_real_words_raise_error(app, text, n_real):
    word = "word" if n_real == 1 else "words"
    with pytest.raises(app.gr.Error, match=rf"The text has {n_real} {word}\. Please enter at least 50 words; "
                                           r"the detector needs more text to give a meaningful result\."):
        app.detect(text)


def test_49_real_words_raise_error_even_with_extra_numbers(app):
    text = words(app.MIN_INPUT_WORDS - 1) + " 1 2 3 4 5 6 7 8 9 10 % – 2,5"
    assert len(text.split()) > app.MIN_INPUT_WORDS                      # enough tokens, but not enough words
    with pytest.raises(app.gr.Error, match=r"The text has 49 words\. Please enter at least 50 words"):
        app.detect(text)


def test_50_real_words_are_accepted(app):
    overall, body, notes = app.detect(words(app.MIN_INPUT_WORDS))
    assert sum(overall.values()) == pytest.approx(1.0)


def test_real_words_include_umlauts_and_eszett(app):
    assert app.real_word_count("Größe Übung ß 42 3,5 % – (z.B.)") == 4   # Größe, Übung, ß, (z.B.)
