"""Shared data and reporting for the behavioral tests (CheckList: MFT, INV, DIR).

Every test records the probabilities it saw in ``RESULTS``. At the end of the run they are
printed as a summary, with the largest change of the top probability over all INV cases.
"""

import json
from pathlib import Path

import pytest

TEXTS_DIR = Path(__file__).parent / "texts"
CLASSES = ["human", "ai_generated", "ai_assisted"]

RESULTS: list[dict] = []


def load_demo_sentences() -> dict[str, list[str]]:
    """The three demo texts from the README GIF, as lists of sentences."""
    return json.loads((TEXTS_DIR / "demo_texts.json").read_text(encoding="utf-8"))["texts"]


def join_sentences(sentences: list[str]) -> str:
    return " ".join(sentences)


@pytest.fixture(scope="session")
def demo_sentences() -> dict[str, list[str]]:
    return load_demo_sentences()


@pytest.fixture(scope="session")
def demo_texts(demo_sentences) -> dict[str, str]:
    return {label: join_sentences(sentences) for label, sentences in demo_sentences.items()}


@pytest.fixture(scope="session")
def inv_texts(demo_texts) -> dict[str, dict]:
    """All INV texts: the demo texts plus the extra texts in inv_texts.json, as {name: {"text", "expected_class"}}."""
    extra = json.loads((TEXTS_DIR / "inv_texts.json").read_text(encoding="utf-8"))["texts"]
    texts = {label: {"text": text, "expected_class": label} for label, text in demo_texts.items()}
    texts.update({name: {"text": entry["text"], "expected_class": entry["expected_class"]} for name, entry in extra.items()})
    return texts


@pytest.fixture
def record(request):
    """Store one result line for the summary: record(kind, case, **numbers)."""

    def _record(kind: str, case: str, **values):
        RESULTS.append({"kind": kind, "case": case, "test": request.node.nodeid, **values})

    return _record


def format_probs(probs: dict[str, float]) -> str:
    return "  ".join(f"{name}={probs[name]:.4f}" for name in CLASSES)


def pytest_terminal_summary(terminalreporter):
    if not RESULTS:
        return
    write = terminalreporter.write_line
    terminalreporter.section("behavioral results")
    for kind in ("MFT", "INV", "DIR"):
        rows = [r for r in RESULTS if r["kind"] == kind]
        if not rows:
            continue
        write(f"--- {kind} ---")
        for r in rows:
            write(f"{r['case']}")
            for key, value in r.items():
                if key in ("kind", "case", "test"):
                    continue
                write(f"    {key}: {format_probs(value) if isinstance(value, dict) else value}")
    inv = [r for r in RESULTS if r["kind"] == "INV"]
    if inv:
        worst = max(inv, key=lambda r: r["top_prob_change"])
        write(f"Largest INV change of the top probability: {worst['top_prob_change']:.6f} ({worst['case']})")
