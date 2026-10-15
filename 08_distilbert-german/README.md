[← 07 · mmBERT-small](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/07_mmbert-small)

# 08 · DistilBERT German

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`distilbert/distilbert-base-german-cased`](https://huggingface.co/distilbert/distilbert-base-german-cased) and uses a maximum sequence length of 512 tokens.

## Setup

A fine-tuned DistilBERT transformer classifier for German text.

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

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.56% accuracy**.

The binary confusion matrix contains:

- **910** true positives
- **908** true negatives
- **6** false positives
- **2** false negatives

## Text Length

Performance varies with text length, with very short texts being the most difficult.

| **Words** | **Samples** | **Accuracy** | **Error** |
| --------- | ----------- | ------------ | --------- |
| ≤70 | 399 | 83.21% | 16.79% |
| 71–150 | 293 | 90.10% | 9.90% |
| 151–300 | 640 | 92.19% | 7.81% |
| 301–500 | 342 | **93.86%** | **6.14%** |
| 501–750 | 653 | 92.50% | 7.50% |
| 750+ | 410 | 91.46% | 8.54% |

The largest performance drop occurs for texts of **70 words or fewer**. Performance is substantially more stable once more text is available.

## How to Run

The trained three-class model is available on [**Hugging Face**](https://huggingface.co/HassanHijazi/german-ai-text-distilbert-german) and can be loaded directly with `transformers`.

To reproduce the training and evaluation locally, place the dataset at `data/hf-dataset-3class-full`, start JupyterLab from the repository root with `jupyter lab`, and run `08_distilbert-german/distilbert_base_german_cased.ipynb` from top to bottom.

A GPU is recommended for training and evaluation.

---

[← 07 · mmBERT-small](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/07_mmbert-small) · [09 · mmBERT-base →](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/09_mmbert-base)