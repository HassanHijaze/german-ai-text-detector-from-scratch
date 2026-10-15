[← 05 · Luna 6 Zero-Shot Baseline](../05_luna-6-baseline/)

# 06 · TF-IDF + Logistic Regression Baseline

A classical machine-learning baseline for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model uses TF-IDF word and bigram features with Logistic Regression.

## Results

| Metric | Result |
|---|---:|
| Accuracy | **91.23%** |
| Macro-F1 | **91.22%** |

| Class | Error |
|---|---:|
| Human | 8.53% |
| AI-generated | 3.40% |
| AI-assisted | 14.38% |

![TF-IDF + Logistic Regression confusion matrix](./results/tfidf-logreg-confmat-3class.svg)

The model performs best on **AI-generated text** and struggles most with **AI-assisted text**, especially when it is classified as Human.

## Human vs AI-generated

| Metric | Result |
|---|---:|
| Samples | 1,826 |
| Accuracy | **98.96%** |

![TF-IDF + Logistic Regression Human vs AI-generated](./results/tfidf-logreg-confmat-human-ai.svg)

This shows that the main difficulty in the full three-class task is distinguishing **Human** from **AI-assisted** writing.

## Text Length

| Words | Accuracy |
|---|---:|
| ≤70 | 76.94% |
| 71–150 | 91.81% |
| 151–300 | 92.97% |
| 301–500 | **94.74%** |
| 501–750 | 94.18% |
| 750+ | 94.39% |

Performance is clearly weaker for very short texts and remains relatively stable from around 150 words onward.

## How to Run

### Use the trained model from Hugging Face

[**HassanHijazi/german-ai-text-tfidf-logreg**](https://huggingface.co/HassanHijazi/german-ai-text-tfidf-logreg)

The trained three-class model can be downloaded directly with `huggingface_hub` and loaded with `joblib`.

### Run the notebook

Make sure the dataset is available at:

```text
data/hf-dataset-3class-full
```

Then start JupyterLab from the repository root:

```bash
jupyter lab
```

Open `06_logreg-baseline/logestic_regression.ipynb` and run the cells in order.

No API key or GPU is required.

---

[← 05 · Luna 6 Zero-Shot Baseline](../05_luna-6-baseline/) · [07 · mmBERT-small →](../07_mmbert-small/)