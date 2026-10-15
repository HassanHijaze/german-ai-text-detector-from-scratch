[← 05 · TF-IDF + Logistic Regression](../05_logreg-baseline/)

# 06 · mmBERT-small

Fine-tuning [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) for the three-class classification task: **Human**, **AI-generated**, and **AI-assisted** text.

## Setup

A fine-tuned transformer classifier based on `jhu-clsp/mmBERT-small`.

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Calibration metrics: Brier score and log loss
- Additional analyses: class-specific errors, text-length performance, and binary Human-vs-AI evaluation
- GPU recommended

## Test results

Evaluation on the held-out test set of **2,737 texts**:

| Split | Accuracy | Brier score | Log loss |
|---|---:|---:|---:|
| Train | 99.89% | 0.0062 | 0.0331 |
| Validation | 95.70% | 0.0701 | 0.1435 |
| Test | **95.94%** | 0.0660 | 0.1369 |

### Class-specific errors

| Class | Error rate |
|---|---:|
| Human | 6.24% |
| AI-generated | 0.99% |
| AI-assisted | 4.94% |

The model is particularly reliable at identifying fully AI-generated text, with only **9 of 912 AI samples** misclassified.

Most remaining errors occur between **Human** and **AI-assisted** text.

### Length analysis

Performance improves substantially with increasing text length.

| Length | Samples | Accuracy |
|---|---:|---:|
| ≤70 | 399 | 87.97% |
| 71–150 | 293 | 95.56% |
| 151–300 | 640 | 95.63% |
| 301–500 | 342 | 98.83% |
| 501–750 | 653 | 98.01% |
| 750+ | 410 | 98.78% |



## Files

```text
06_mmBERT-small/
├── artifacts/
├── results/
│   ├── mmbert-confmat-3class.svg
│   ├── mmbert-confmat-human-ai.svg
│   └── mmbert-small-length-bias-3class.csv
├── mmBERT_small.ipynb
└── README.md
```

## Run

This stage requires the dataset prepared in [04 · HF dataset](../04_hf-dataset/).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run all cells in:

```text
mmBERT_small.ipynb
```

<div align="center">

[← 05 · TF-IDF + Logistic Regression](../05_logreg-baseline/) · [Project index](../README.md) · [07 · DistilBERT →](../07_distilbert-german/)

</div>