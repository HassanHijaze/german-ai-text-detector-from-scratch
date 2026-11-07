[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)

# GeLECTRA-large

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`deepset/gelectra-large`](https://huggingface.co/deepset/gelectra-large) and uses a maximum sequence length of 512 tokens.

## Setup

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and binary Human-vs-AI evaluation
- GPU recommended

## Results

| **Split** | **Accuracy** |
| --------- | ------------ |
| **Test**  | **96.89%** (macro-F1 **96.91%**) |

| **True class** | **Correct** | **Error** |
| -------------- | ----------- | --------- |
| Human          | 95.30%      | 4.70%     |
| AI-generated   | **98.46%**  | **1.54%** |
| AI-assisted    | 96.93%      | 3.07%     |

The model performs very strongly across all three classes.

Human-written text has the highest number of errors, with **43 of 914 Human samples** misclassified.

Fully AI-generated text is detected particularly reliably, with only **14 of 912 AI samples** misclassified.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.62% accuracy**.

The binary confusion matrix contains:

- **906** true positives
- **913** true negatives
- **1** false positive
- **6** false negatives

## Text Length

Performance improves substantially as more text becomes available.

| **Words** | **Samples** | **Accuracy** |
| --------- | ----------- | ------------ |
| ≤70       | 399         | 90.73%       |
| 71–150    | 293         | 97.27%       |
| 151–300   | 640         | 97.66%       |
| 301–500   | 342         | 98.25%       |
| 501–750   | 653         | 97.86%       |
| 751+      | 410         | **98.78%**   |

The largest performance drop occurs for texts of **70 words or fewer**. For longer texts, accuracy remains consistently high at approximately **97–99%**.

Run: `python scripts/run.py gelectra-large` (from `07_fine-tuning/`) · [model on Hugging Face](https://huggingface.co/HassanHijazi/german-ai-text-gelectra-large)

---

[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)