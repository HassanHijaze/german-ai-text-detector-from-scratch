[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)

# mmBERT-small

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) and uses a maximum sequence length of 512 tokens.

## Setup

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and binary Human-vs-AI evaluation
- GPU recommended

## Results

| **Split**  | **Accuracy** |
| ---------- | ------------ |
| Train      | 99.89% |
| Validation | 95.70% |
| **Test**   | **95.94%** (macro-F1 **95.95%**) |

| **True class** | **Correct** | **Error** |
| -------------- | ----------- | --------- |
| Human          | 93.76%      | 6.24% |
| AI-generated   | **99.01%**  | **0.99%** |
| AI-assisted    | 95.06%      | 4.94% |

The model performs particularly well on **AI-generated text**, with only **9 of 912 samples** misclassified.

Most remaining errors occur between **Human** and **AI-assisted** text.

## Text Length

Performance improves substantially as more text becomes available.

| **Words** | **Samples** | **Accuracy** |
| --------- | ----------- | ------------ |
| ≤70       | 399         | 87.97% |
| 71–150    | 293         | 95.56% |
| 151–300   | 640         | 95.63% |
| 301–500   | 342         | 98.83% |
| 501–750   | 653         | 98.01% |
| 751+      | 410         | **98.78%** |

The largest performance drop occurs for texts of **70 words or fewer**. From around 150 words onward, accuracy remains consistently high at approximately **96–99%**.

Run: `python scripts/run.py mmbert-small` (from `07_fine-tuning/`) · [model on Hugging Face](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-small)

---

[← Fine-tuning overview](../../README.md) · [Project index](../../../README.md)