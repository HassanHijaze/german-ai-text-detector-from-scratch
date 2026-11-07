[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)

# DistilBERT German

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`distilbert/distilbert-base-german-cased`](https://huggingface.co/distilbert/distilbert-base-german-cased) and uses a maximum sequence length of 512 tokens.

## Setup

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and binary Human-vs-AI evaluation
- GPU recommended

## Results

| **Split** | **Accuracy** |
| ---------- | ------------ |
| **Test** | **90.83%** (macro-F1 **90.82%**) |

| **True class** | **Correct** | **Error** | **Most common mistake** |
| -------------- | ----------- | --------- | ----------------------- |
| Human | 80.09% | 19.91% | called AI-assisted (178 of 914) |
| AI-generated | **97.92%** | **2.08%** | called AI-assisted (19 of 912) |
| AI-assisted | 94.51% | 5.49% | called Human (37 of 911) |

The model performs particularly well on fully AI-generated text, with only **19 of 912 AI samples** misclassified.

Human-written text is the most difficult class, with **182 of 914 Human samples** classified as either AI-generated or AI-assisted.

## Text Length

Performance varies with text length, with very short texts being the most difficult.

| **Words** | **Samples** | **Accuracy** | **Error** |
| --------- | ----------- | ------------ | --------- |
| ≤70 | 399 | 83.21% | 16.79% |
| 71–150 | 293 | 90.10% | 9.90% |
| 151–300 | 640 | 92.19% | 7.81% |
| 301–500 | 342 | **93.86%** | **6.14%** |
| 501–750 | 653 | 92.50% | 7.50% |
| 751+ | 410 | 91.46% | 8.54% |

The largest performance drop occurs for texts of **70 words or fewer**. Performance is substantially more stable once more text is available.

Run: `python scripts/run.py distilbert-german` (from `07_fine-tuning/`) · [model on Hugging Face](https://huggingface.co/HassanHijazi/german-ai-text-distilbert-german)

---

[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)