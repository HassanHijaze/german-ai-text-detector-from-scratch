[← 10 · OOD Model Comparison](../10_ood-model-comparison/)

# 11 · Deployment

A small web app that classifies German text as **Human**, **AI-generated** or **AI-assisted**, packaged as a Docker container.

It uses the fine-tuned **mmBERT-base** model, one of the project's two strongest models, recommended for its robustness to unseen human-text domains (see [09 · Model Comparison](../09_model-comparison/) and [10 · OOD Model Comparison](../10_ood-model-comparison/)).

## Features

- **Web interface** built with Gradio: paste a text, click **Detect**, see the result.
- **Full text analysed in windows**: the text is split into sentences and classified in overlapping windows of about 100 words, so the whole text is analysed, however long it is. Each sentence gets the average of all windows that contain it (windows in which it sits near the centre count more).
- **Coloured parts**: neighbouring sentences with the same class form one coloured part, so human and AI parts of a text are coloured differently:
  - light blue: Human
  - light yellow: AI-assisted
  - light red: AI-generated
- **Percentages on hover**: hovering over a part shows its probability for each class.
- **Share bar**: a bar above the text shows the share of the text (in words) for each class.
- **Overall scores** for all three classes: a length-weighted average over all sentences.
- **Calibrated probabilities**: temperature scaling with the model's `detector-config.json`, if the model has one; otherwise the raw (uncalibrated) probabilities are used.
- **Warnings** for short texts (under 70 words), where results are much less reliable, and when a window is longer than 512 tokens.
- **Runs offline**: the model is downloaded once, when the image is built.
- **API included**: the app can also be called from other programs.

## Quick start

Requires [Docker](https://www.docker.com/products/docker-desktop/).

```bash
cd 11_deployment
docker build -t german-ai-text-detector .
docker run -p 7860:7860 german-ai-text-detector
```

Then open **http://localhost:7860**.

The first build takes a few minutes. The image is about 2.5–3 GB, mostly PyTorch and the model. The app runs on the CPU; no GPU is needed.

To stop the app, press `Ctrl + C`.

## Using the API

While the container is running, the detector can also be called from Python:

```bash
pip install gradio_client
```

```python
from gradio_client import Client

client = Client("http://localhost:7860/")
scores, highlighted_html, notes = client.predict("Ihr deutscher Text …", api_name="/detect")
print(scores)
```

`scores` contains the length-weighted probability of each class over the whole text.

## Configuration

| Setting | How | Default |
|---|---|---|
| Model | build argument `MODEL_ID` | `HassanHijazi/german-ai-text-mmbert-base` |
| Port | environment variable `PORT` | `7860` |

For example, to build the image with GeLECTRA-large instead:

```bash
docker build --build-arg MODEL_ID=HassanHijazi/german-ai-text-gelectra-large -t german-ai-text-detector .
```

Other models only get calibrated probabilities if their Hugging Face repo contains `detector-config.json` (DistilBERT German, for example, does not). The text on the page always names mmBERT-base, whichever model is built in.

## Files

```text
11_deployment/
├── Dockerfile          builds the image and downloads the model
├── app.py              the Gradio web app
├── requirements.txt    Python packages (PyTorch is installed in the Dockerfile)
├── .dockerignore
└── README.md
```

## Limitations

- The model was trained on **German academic texts**. Results on other kinds of text (social media, fiction, other languages) may be less reliable.
- **Short texts** (under 70 words) are much harder to classify correctly.
- **Boundaries** between parts are accurate to about one sentence, and insertions of only one or two sentences may not be detected.
- The result is a **probabilistic estimate, not proof**. It must not be used on its own to accuse, grade or make other high-stakes decisions about people.

<div align="center">

[← 10 · OOD Model Comparison](../10_ood-model-comparison/) · [Project index](../README.md)

</div>
