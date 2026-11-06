[← 14 · OOD Model Comparison](../14_ood-model-comparison/)

# 15 · Deployment

A small web app that classifies German text as **Human**, **AI-generated** or **AI-assisted**, packaged as a Docker container.

It uses the fine-tuned **mmBERT-base** model, the project's most robust model on new kinds of text (see [13 · Model Comparison](../13_model-comparison/) and [14 · OOD Model Comparison](../14_ood-model-comparison/)).

## Features

- **Web interface** built with Gradio: paste a text, click **Detect**, see the result.
- **Probabilities** for all three classes, shown as bars.
- **Coloured text**: the analysed text is shown on a soft background in the colour of the predicted class:
  - light blue: Human
  - light yellow: AI-assisted
  - light red: AI-generated
- **Warnings** for short texts (under 70 words), where results are much less reliable, and for long texts, where only the first 512 tokens are analysed.
- **Runs offline**: the model is downloaded once, when the image is built.
- **API included**: the app can also be called from other programs.

## Quick start

Requires [Docker](https://www.docker.com/products/docker-desktop/).

```bash
cd 15_deployment
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

`scores` contains the predicted class and the probability of each class.

## Configuration

| Setting | How | Default |
|---|---|---|
| Model | build argument `MODEL_ID` | `HassanHijazi/german-ai-text-mmbert-base` |
| Port | environment variable `PORT` | `7860` |

For example, to build the image with GeLECTRA-large instead:

```bash
docker build --build-arg MODEL_ID=HassanHijazi/german-ai-text-gelectra-large -t german-ai-text-detector .
```

## Files

```text
15_deployment/
├── Dockerfile          builds the image and downloads the model
├── app.py              the Gradio web app
├── requirements.txt    Python packages (PyTorch is installed in the Dockerfile)
├── .dockerignore
└── README.md
```

## Limitations

- The model was trained on **German academic texts**. Results on other kinds of text (social media, fiction, other languages) may be less reliable.
- **Short texts** (under 70 words) are much harder to classify correctly.
- Only the **first 512 tokens** of a text are analysed.
- The result is a **probabilistic estimate, not proof**. It must not be used on its own to accuse, grade or make other high-stakes decisions about people.

<div align="center">

[← 14 · OOD Model Comparison](../14_ood-model-comparison/) · [Project index](../README.md)

</div>
