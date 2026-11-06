"""Gradio web app for the German AI text detector.

Classifies a German text as Human, AI-generated or AI-assisted with the
fine-tuned mmBERT-base model. The model folder is set with MODEL_DIR
(inside the Docker image: /app/model).
"""

import html
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
PORT = int(os.getenv("PORT", "7860"))
MAX_LENGTH = 512        # same limit as in training
MIN_WORDS = 70          # below this, accuracy drops clearly (see step 13 / length analysis)

# Label order of the trained model: 0 = human, 1 = AI-generated, 2 = AI-assisted
LABELS = ["Human", "AI-generated", "AI-assisted"]

# Soft background colours for the highlighted text (background, border)
COLORS = {
    "Human": ("#E3F0FF", "#9CC3F5"),          # light blue
    "AI-generated": ("#FDE4E4", "#F2A7A7"),   # light red
    "AI-assisted": ("#FFF6D6", "#F0D27A"),    # light yellow
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()


def detect(text: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter a German text.")

    encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0].tolist()

    notes = []
    words = len(text.split())
    if words < MIN_WORDS:
        notes.append(f"⚠️ Short text ({words} words). Results for texts under {MIN_WORDS} words are much less reliable.")
    if len(tokenizer(text)["input_ids"]) > MAX_LENGTH:
        notes.append(f"ℹ️ Long text. Only the first {MAX_LENGTH} tokens were analysed.")

    scores = dict(zip(LABELS, probs))
    return scores, highlight(text, scores), "\n\n".join(notes)


def highlight(text: str, scores: dict) -> str:
    """Show the text on a soft background in the colour of the predicted class."""
    label = max(scores, key=scores.get)
    background, border = COLORS[label]
    body = html.escape(text).replace("\n", "<br>")
    return (
        f'<div style="background:{background}; border:1px solid {border}; border-radius:10px; '
        f'padding:16px; color:#1f2937; line-height:1.6;">'
        f'<div style="font-weight:600; margin-bottom:8px;">{label} · {scores[label]:.0%}</div>'
        f"{body}</div>"
    )


with gr.Blocks(title="German AI Text Detector") as demo:
    gr.Markdown(
        "# German AI Text Detector\n"
        "Classifies German text as **Human**, **AI-generated** or **AI-assisted** "
        "(human text rewritten by an AI). Model: fine-tuned mmBERT-base, trained on German academic texts.\n\n"
        "The analysed text is shown below the result in "
        "<span style='background:#E3F0FF;color:#1f2937;padding:1px 6px;border-radius:4px'>blue</span> for Human, "
        "<span style='background:#FDE4E4;color:#1f2937;padding:1px 6px;border-radius:4px'>red</span> for AI-generated and "
        "<span style='background:#FFF6D6;color:#1f2937;padding:1px 6px;border-radius:4px'>yellow</span> for AI-assisted."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(label="German text", lines=12, placeholder="Paste a German text here…")
            with gr.Row():
                clear_btn = gr.ClearButton(text_in)
                detect_btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=2):
            result = gr.Label(label="Prediction", num_top_classes=3)
            notes = gr.Markdown()

    highlighted = gr.HTML()

    gr.Examples([[EXAMPLE]], inputs=text_in)
    gr.Markdown(
        "**Note:** The output is a probabilistic estimate, not proof. Do not use it on its own "
        "to accuse, grade or make other high-stakes decisions about people. The model works best "
        "on German academic texts of 70 words or more."
    )

    detect_btn.click(detect, inputs=text_in, outputs=[result, highlighted, notes], api_name="detect")
    text_in.submit(detect, inputs=text_in, outputs=[result, highlighted, notes])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
