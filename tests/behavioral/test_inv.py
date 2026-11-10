"""INV (Invariance test): changes that are irrelevant to authorship must not change the prediction.

Each perturbation is applied to each INV text: the three demo texts and the extra texts in
data/inv_texts.json (a human test text with quotes, years and a cited author, so that the quote
swap, number shift and name swap actually run). The original text must be predicted as its
expected class, and the predicted class must stay the same after the change.
The change of the top probability (the probability of the class predicted for the
original text) is recorded for every case; the largest one is printed at the end of the run.

Perturbations that need something the text does not contain (typographic quotes,
numbers, a cited name) are skipped for that text, because they would leave it unchanged.
"""

import re

import pytest

pytestmark = pytest.mark.behavioral

CASES = ["human", "ai_assisted", "ai_generated", "human_test_id_327"]


def double_spaces(text: str) -> str | None:
    """Two spaces instead of one after every sentence end."""
    return re.sub(r"(?<=[.!?]) (?=\S)", "  ", text)


def trailing_newline(text: str) -> str | None:
    return text + "\n"


def swap_quotes(text: str) -> str | None:
    """German quotes „…“ become English quotes “…” and the other way round."""
    pattern = re.compile(r"„([^„“”]*)“|“([^„“”]*)”")
    if not pattern.search(text):
        return None
    return pattern.sub(lambda m: f"“{m.group(1)}”" if m.group(1) is not None else f"„{m.group(2)}“", text)


def shift_numbers(text: str) -> str | None:
    """Years (1900-2099) move by two (2019 -> 2021), other numbers by one."""
    if not re.search(r"\d", text):
        return None

    def shift(match):
        number = int(match.group())
        return str(number + 2 if 1900 <= number <= 2099 else number + 1)

    return re.sub(r"\d+", shift, text)


CITED_NAME = re.compile(r"\b[A-ZÄÖÜ][a-zäöüß]+(?=(?: et al\.)?,? \(?\d{4})")  # "Esser 1999", "Esser (1999)"


def swap_cited_name(text: str) -> str | None:
    """The first cited author name becomes "Müller" (or "Schmidt" if it already is "Müller")."""
    match = CITED_NAME.search(text)
    if not match:
        return None
    name = match.group()
    return re.sub(rf"\b{name}\b", "Schmidt" if name == "Müller" else "Müller", text)


PERTURBATIONS = {
    "double_spaces": (double_spaces, ""),
    "trailing_newline": (trailing_newline, ""),
    "swap_quotes": (swap_quotes, "typographic quotes"),
    "shift_numbers": (shift_numbers, "numbers or years"),
    "swap_cited_name": (swap_cited_name, "cited person's name"),
}


@pytest.mark.parametrize("perturbation", list(PERTURBATIONS))
@pytest.mark.parametrize("case", CASES)
def test_irrelevant_change_keeps_the_prediction(detector, inv_texts, record, case, perturbation):
    perturb, needs = PERTURBATIONS[perturbation]
    original, expected = inv_texts[case]["text"], inv_texts[case]["expected_class"]
    changed = perturb(original)
    if changed is None:
        pytest.skip(f"the {case} text contains no {needs}")
    assert changed != original

    before = detector.probabilities(original)
    after = detector.probabilities(changed)
    top_before = max(before, key=before.get)
    top_after = max(after, key=after.get)
    change = abs(after[top_before] - before[top_before])
    record("INV", f"{case} + {perturbation}", original=before, perturbed=after,
           top_class=top_before, top_prob_change=change)

    assert top_before == expected, f"the original {case} text is predicted as {top_before}, not {expected}: {before}"
    assert top_after == top_before, (
        f"{perturbation} changed the prediction of the {case} text from {top_before} to {top_after}: "
        f"before {before}, after {after}"
    )
