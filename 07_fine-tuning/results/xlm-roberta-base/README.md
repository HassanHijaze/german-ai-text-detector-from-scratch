[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)

# XLM-RoBERTa base

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) and uses a maximum sequence length of 512 tokens.

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
| **Test**  | **91.09%** (macro-F1 **91.03%**) |

| **True class** | **Correct** | **Error** |
| -------------- | ----------- | --------- |
| Human          | 78.34%      | 21.66%    |
| AI-generated   | **99.12%**  | **0.88%** |
| AI-assisted    | 95.83%      | 4.17%     |

The model performs particularly well on **AI-generated text**, with only **8 of 912 samples** misclassified.

Human-written text is the most difficult class, with **198 of 914 Human samples** misclassified, primarily as AI-assisted text.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.56% accuracy**.

The binary confusion matrix contains:

- **912** true positives
- **906** true negatives
- **8** false positives
- **0** false negatives

## Text Length

Performance varies with text length, with very short texts being the most difficult.

| **Words** | **Samples** | **Accuracy** |
| --------- | ----------- | ------------ |
| ≤70       | 399         | 82.96%       |
| 71–150    | 293         | 91.13%       |
| 151–300   | 640         | 91.72%       |
| 301–500   | 342         | **94.44%**   |
| 501–750   | 653         | 93.57%       |
| 751+      | 410         | 91.22%       |

The largest performance drop occurs for texts of **70 words or fewer**. Performance improves substantially for longer texts, reaching its highest accuracy for samples between **301 and 500 words**.

Run: `python scripts/run.py xlm-roberta-base` (from `07_fine-tuning/`) · [model on Hugging Face](https://huggingface.co/HassanHijazi/german-ai-text-xlm-roberta-base)

---

[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)