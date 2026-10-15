[← 10 · XLM-RoBERTa base](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/10_xlm-roberta-base)

# 11 · GeLECTRA-large

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`deepset/gelectra-large`](https://huggingface.co/deepset/gelectra-large) and uses a maximum sequence length of 512 tokens.

## Setup

A fine-tuned GeLECTRA-large transformer classifier for German text.

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
| 750+      | 410         | **98.78%**   |

The largest performance drop occurs for texts of **70 words or fewer**. For longer texts, accuracy remains consistently high at approximately **97–99%**.

## How to Run

The trained three-class model is available on [**Hugging Face**](https://huggingface.co/HassanHijazi/german-ai-text-gelectra-large) and can be loaded directly with `transformers`.

To reproduce the training and evaluation locally, place the dataset at `data/hf-dataset-3class-full`, start JupyterLab from the repository root with `jupyter lab`, and run `11_gelectra-large/Gelectra_large.ipynb` from top to bottom.

A GPU is recommended for training and evaluation.

---

[← 10 · XLM-RoBERTa base](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/10_xlm-roberta-base) · [12 · Learning Curves →](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/12_learning-curves)