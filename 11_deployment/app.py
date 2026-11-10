"""Gradio web app for the German AI text detector.

Classifies German text as Human, AI-generated or AI-assisted with the
fine-tuned mmBERT-base model. The model folder is set with MODEL_DIR
(inside the Docker image: /app/model).

Texts are analysed sentence by sentence, so human and AI writing can be
told apart even inside one paragraph:
  1. The text is split into sentences (paragraph breaks also end a sentence).
  2. Overlapping windows of about WINDOW_WORDS words slide over the sentences,
     moving STRIDE_WORDS words at a time, and every window is classified.
  3. Each sentence gets a weighted average of all windows that contain it; windows
     in which the sentence sits near the centre count more.
  4. Neighbouring sentences with the same class are shown as one coloured span
     in the colour of the most likely class. The original text is shown unchanged.

Probabilities are calibrated with the temperature stored in the model's
detector-config.json (temperature scaling, fitted on the validation set).
If the file is missing, the raw (uncalibrated) probabilities are used.
"""

import html
import json
import math
import os
import re
from pathlib import Path

import gradio as gr
import torch
from langdetect import DetectorFactory, LangDetectException
from langdetect import detect as detect_language
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DetectorFactory.seed = 0  # make language detection deterministic

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
PORT = int(os.getenv("PORT", "7860"))

MAX_LENGTH = 512        # model input limit in tokens (same as in training)
MIN_INPUT_WORDS = 50    # shorter inputs (counting only real words) are rejected with an error
MIN_WORDS = 70          # below this, accuracy drops clearly (see the length analysis)
MAX_WORDS = 3000        # longer inputs are rejected with an error
WINDOW_WORDS = 100      # each window holds at least this many words (if the text is long enough)
STRIDE_WORDS = 20       # a new window starts every ~20 words, so windows overlap strongly
BATCH_SIZE = 8

# Label order of the trained model: 0 = human, 1 = AI-generated, 2 = AI-assisted
LABELS = ["Human", "AI-generated", "AI-assisted"]

# Names for the languages langdetect reports most often (other codes are shown as they are)
LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "it": "Italian", "nl": "Dutch",
    "pt": "Portuguese", "pl": "Polish", "da": "Danish", "sv": "Swedish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "tr": "Turkish", "ru": "Russian", "ar": "Arabic",
}

# Soft background colours (background, border)
COLORS = {
    "Human": ("#E3F0FF", "#9CC3F5"),          # light blue
    "AI-generated": ("#FDE4E4", "#F2A7A7"),   # light red
    "AI-assisted": ("#FFF6D6", "#F0D27A"),    # light yellow
}

# Abbreviations that end with a dot but do not end a sentence
ABBREVIATIONS = {
    "z.b.", "b.", "d.h.", "h.", "u.a.", "a.", "bzw.", "vgl.", "ca.", "dr.", "prof.", "nr.",
    "s.", "usw.", "etc.", "abb.", "tab.", "bspw.", "ggf.", "inkl.", "evtl.", "sog.", "u.",
    "z.", "d.", "jh.", "mio.", "mrd.", "hrsg.", "ebd.", "f.", "ff.", "st.", "str.",
}

EXAMPLE = (
    "Die vorliegende Untersuchung analysiert die Auswirkungen digitaler Lernplattformen "
    "auf die Motivation von Schülerinnen und Schülern der Sekundarstufe. Auf Grundlage "
    "einer Befragung an zwölf Schulen zeigt sich, dass vor allem die unmittelbare "
    "Rückmeldung und die freie Zeiteinteilung positiv bewertet werden, während fehlender "
    "persönlicher Kontakt zu Lehrkräften als Nachteil wahrgenommen wird. Die Ergebnisse "
    "deuten darauf hin, dass digitale Angebote den Präsenzunterricht ergänzen, aber nicht "
    "vollständig ersetzen sollten. Weitere Studien mit größeren Stichproben und "
    "längeren Beobachtungszeiträumen sind notwendig, um diese Befunde abzusichern."
)


# ---------- Model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()


def load_temperature(model_dir: str) -> float:
    """Read the calibration temperature from detector-config.json (1.0 = no calibration)."""
    config_path = Path(model_dir) / "detector-config.json"
    if not config_path.exists():
        print("detector-config.json not found: using uncalibrated probabilities.")
        return 1.0
    temperature = float(json.loads(config_path.read_text(encoding="utf-8")).get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError(f"Invalid temperature in {config_path}: {temperature}")
    print(f"Using calibration temperature {temperature:.4f}")
    return temperature


TEMPERATURE = load_temperature(MODEL_DIR)


def predict(texts: list[str]) -> list[list[float]]:
    """Calibrated class probabilities for each text."""
    probs = []
    for start in range(0, len(texts), BATCH_SIZE):
        encoded = tokenizer(texts[start:start + BATCH_SIZE], truncation=True, max_length=MAX_LENGTH,
                            padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**encoded).logits / TEMPERATURE
            probs.extend(torch.softmax(logits, dim=-1).tolist())
    return probs


# ---------- Splitting (works on character positions, so the original text is kept) ----------
def word_count(text: str) -> int:
    return len(text.split())


def real_word_count(text: str) -> int:
    """Number of tokens with at least one letter (ä, ö, ü, ß included); numbers and symbols alone do not count."""
    return sum(any(char.isalpha() for char in token) for token in text.split())


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every non-empty paragraph; paragraphs are separated by empty lines."""
    spans, start = [], 0
    for match in re.finditer(r"\n\s*\n", text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def sentence_starts(text: str, start: int, end: int) -> list[int]:
    """Positions inside (start, end) where a new sentence begins."""
    starts = []
    for match in re.finditer(r"[.!?…][\"'»«“”)\]]*\s+", text[start:end]):
        before = text[start:start + match.start() + 1].split()
        last_word = before[-1].lower() if before else ""
        if last_word in ABBREVIATIONS or re.fullmatch(r"\d+\.", last_word):
            continue  # abbreviation or ordinal number such as "3."
        starts.append(start + match.end())
    return [pos for pos in starts if pos < end]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every sentence, without surrounding whitespace."""
    spans = []
    for p_start, p_end in paragraph_spans(text):
        bounds = [p_start] + sentence_starts(text, p_start, p_end) + [p_end]
        for s, e in zip(bounds, bounds[1:]):
            chunk = text[s:e]
            if chunk.strip():
                s += len(chunk) - len(chunk.lstrip())
                e -= len(chunk) - len(chunk.rstrip())
                spans.append((s, e))
    return spans


def make_windows(words: list[int]) -> list[tuple[int, int]]:
    """Overlapping windows of consecutive sentences, as (first, last + 1) sentence indices."""
    n, windows, i = len(words), [], 0
    while i < n:
        j, count = i, 0
        while j < n and count < WINDOW_WORDS:
            count += words[j]
            j += 1
        start = i
        if j == n:  # last window: extend backwards so it is not too short
            while start > 0 and count < WINDOW_WORDS:
                start -= 1
                count += words[start]
        windows.append((start, j))
        if j == n:
            break
        step = 0  # move forward by about STRIDE_WORDS words
        while i < n - 1 and step < STRIDE_WORDS:
            step += words[i]
            i += 1
    return windows


# ---------- Input checks ----------
def language_warning(text: str) -> str | None:
    """A warning if the text does not look German. Only called for texts of at least
    MIN_INPUT_WORDS real words, since language detection is unreliable on shorter texts."""
    try:
        language = detect_language(text)
    except LangDetectException:
        return None  # detection failed: no warning
    if language == "de":
        return None
    name = LANGUAGE_NAMES.get(language, language)
    return (f"⚠️ This text doesn't look German (detected: {name}). The model was trained only on "
            "German texts, so the result may be unreliable.")


# ---------- Output ----------
def part_label(scores: dict) -> str:
    """The most likely class."""
    return max(scores, key=scores.get)


def render(text: str, spans: list[tuple[int, int, dict]]) -> str:
    """The original text, unchanged, with each span on the background colour of its class."""
    pieces, cursor = [], 0
    for start, end, scores in spans:
        pieces.append(html.escape(text[cursor:start]))  # text between spans, unchanged
        label = part_label(scores)
        background, border = COLORS[label]
        tooltip = " · ".join(f"{name} {scores[name]:.0%}" for name in sorted(scores, key=scores.get, reverse=True))
        pieces.append(
            f'<span title="{html.escape(label)}: {html.escape(tooltip)}" style="background:{background}; '
            f'box-shadow:0 0 0 1px {border}; border-radius:4px; padding:1px 2px; '
            f'-webkit-box-decoration-break:clone; box-decoration-break:clone;">'
            f"{html.escape(text[start:end])}</span>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def summary_bar(shares: dict) -> str:
    """Stacked bar with the share of words per class."""
    segments, legend = [], []
    for label in LABELS:
        share = shares.get(label, 0.0)
        if share <= 0:
            continue
        background, border = COLORS[label]
        segments.append(f'<div style="width:{share:.2%}; background:{background}; border-right:1px solid {border};"></div>')
        legend.append(
            f'<span style="margin-right:14px;"><span style="display:inline-block; width:10px; height:10px; '
            f'background:{background}; border:1px solid {border}; border-radius:2px; margin-right:5px;"></span>'
            f"{label} {share:.0%}</span>"
        )
    return (
        '<div style="display:flex; height:14px; border-radius:7px; overflow:hidden; border:1px solid #d1d5db; '
        f'margin-bottom:6px;">{"".join(segments)}</div>'
        f'<div style="font-size:0.9em; margin-bottom:14px;">{"".join(legend)}</div>'
    )


def detect(text: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter a German text.")
    n_real = real_word_count(text)
    if n_real < MIN_INPUT_WORDS:
        raise gr.Error(f"The text has {n_real} word{'' if n_real == 1 else 's'}. Please enter at least "
                       f"{MIN_INPUT_WORDS} words; the detector needs more text to give a meaningful result.")
    n_words = word_count(text)
    if n_words > MAX_WORDS:
        raise gr.Error(f"The text has {n_words:,} words. Please enter at most {MAX_WORDS:,} words.")

    sentences = sentence_spans(text)
    words = [word_count(text[s:e]) for s, e in sentences]
    windows = make_windows(words)
    window_texts = [text[sentences[a][0]:sentences[b - 1][1]] for a, b in windows]
    window_probs = predict(window_texts)

    # Each sentence: weighted average of all windows that contain it. A window counts
    # more the closer the sentence is to its centre (measured in words).
    positions, cursor = [], 0
    for w in words:
        positions.append(cursor + w / 2)  # word position of each sentence's centre
        cursor += w
    sums = [[0.0] * len(LABELS) for _ in sentences]
    weights = [0.0] * len(sentences)
    for (a, b), probs in zip(windows, window_probs):
        w_start = positions[a] - words[a] / 2
        w_end = positions[b - 1] + words[b - 1] / 2
        centre, half = (w_start + w_end) / 2, max((w_end - w_start) / 2, 1)
        for i in range(a, b):
            weight = max(1 - abs(positions[i] - centre) / half, 0.05)
            sums[i] = [x + weight * p for x, p in zip(sums[i], probs)]
            weights[i] += weight
    sentence_scores = [dict(zip(LABELS, [x / wt for x in row])) for row, wt in zip(sums, weights)]

    # Join neighbouring sentences with the same class into one span
    spans = []
    for (start, end), w, scores in zip(sentences, words, sentence_scores):
        label = part_label(scores)
        if spans and spans[-1]["label"] == label:
            span = spans[-1]
            span["end"] = end
            span["words"] += w
            for name in LABELS:
                span["weighted"][name] += w * scores[name]
        else:
            spans.append({"start": start, "end": end, "label": label, "words": w,
                          "weighted": {name: w * scores[name] for name in LABELS}})
    display = [(sp["start"], sp["end"], {n: v / max(sp["words"], 1) for n, v in sp["weighted"].items()})
               for sp in spans]

    # Overall result and shares, weighted by the number of words
    total = sum(words)
    overall = {name: sum(w * s[name] for w, s in zip(words, sentence_scores)) / total for name in LABELS}
    shares: dict = {}
    for sp in spans:
        shares[sp["label"]] = shares.get(sp["label"], 0.0) + sp["words"] / total

    notes = []
    warning = language_warning(text)
    if warning:
        notes.append(warning)
    if total < MIN_WORDS:
        notes.append(f"⚠️ Short text ({total} words). Results for texts under {MIN_WORDS} words are much less reliable.")
    if len(windows) > 1:
        notes.append(f"ℹ️ The text was analysed in {len(windows)} overlapping windows of about {WINDOW_WORDS} words, "
                     "so the colour can change inside a paragraph. Boundaries are approximate, and an insertion "
                     "of only one or two sentences may not be detected. Hover over a part to see its percentages.")
    if any(len(tokenizer(t)["input_ids"]) > MAX_LENGTH for t in window_texts):
        notes.append(f"ℹ️ At least one window is longer than {MAX_LENGTH} tokens; only its beginning was analysed.")

    body = (
        '<div style="border:1px solid #d1d5db; border-radius:10px; padding:16px; color:#1f2937; '
        'background:#ffffff; line-height:1.9;">'
        f"{summary_bar(shares)}"
        f'<div style="white-space:pre-wrap;">{render(text, display)}</div></div>'
    )
    return overall, body, "\n\n".join(notes)


# ---------- Interface ----------
def legend_chip(label: str, text: str) -> str:
    background, _ = COLORS[label]
    return f"<span style='background:{background};color:#1f2937;padding:1px 6px;border-radius:4px'>{text}</span>"


with gr.Blocks(title="German AI Text Detector") as demo:
    gr.Markdown(
        "# German AI Text Detector\n"
        "Classifies German text as **Human**, **AI-generated** or **AI-assisted** "
        "(human text rewritten by an AI). Model: fine-tuned mmBERT-base, trained on German academic texts.\n\n"
        "Texts are analysed sentence by sentence, so the colour can change even inside a paragraph. Each part is shown in "
        f"{legend_chip('Human', 'blue')} for Human, {legend_chip('AI-generated', 'red')} for AI-generated "
        f"and {legend_chip('AI-assisted', 'yellow')} for AI-assisted."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(label="German text", lines=12,
                                 placeholder=f"Paste a German text here ({MIN_INPUT_WORDS}–{MAX_WORDS:,} words)…")
            with gr.Row():
                clear_btn = gr.ClearButton(text_in)
                detect_btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=2):
            result = gr.Label(label="Overall prediction (weighted by length)", num_top_classes=3)
            notes = gr.Markdown()

    highlighted = gr.HTML()

    gr.Examples([[EXAMPLE]], inputs=text_in)
    gr.Markdown(
        "**Note:** The output is a probabilistic estimate, not proof. Do not use it on its own "
        "to accuse, grade or make other high-stakes decisions about people. The model works best "
        "on German academic texts of 70 words or more. Boundaries between parts are approximate, and very "
        "short insertions (one or two sentences) may not be detected."
    )

    detect_btn.click(detect, inputs=text_in, outputs=[result, highlighted, notes], api_name="detect")
    text_in.submit(detect, inputs=text_in, outputs=[result, highlighted, notes])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)