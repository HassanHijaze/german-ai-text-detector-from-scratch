[← 09 · XLM-RoBERTa-base](../09_xlm-roberta-base/)

# 10 · GeLECTRA-large

Fine-tuning a German GeLECTRA-large model for the three-class 
classification task: **Human**, **AI-generated**, and **AI-assisted** 
text.

## Setup

A fine-tuned GeLECTRA-large transformer classifier for German text.

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and 
binary Human-vs-AI evaluation
- GPU recommended

## Test results

Evaluation on the held-out test set of **2,737 texts**:

| Accuracy | Macro-F1 | Human error | AI-generated error | AI-assisted error |
|---:|---:|---:|---:|---:|
| **96.89%** | **96.91%** | 4.70% | 1.54% | 3.07% |

### Class-specific errors

| Class | Error rate |
|---|---:|
| Human | 4.70% |
| AI-generated | 1.54% |
| AI-assisted | 3.07% |

The model performs very strongly across all three classes.

Human-written text has the highest number of errors, with **43 of 914 
Human samples** misclassified.

Fully AI-generated text is detected particularly reliably, with only **14 
of 912 AI samples** misclassified.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.62% 
accuracy**.

The binary confusion matrix contains:

- **906** true positives
- **913** true negatives
- **1** false positive
- **6** false negatives

### Length analysis

Performance improves substantially with increasing text length.

| Length | Samples | Accuracy |
|---|---:|---:|
| ≤70 | 399 | 90.73% |
| 71–150 | 293 | 97.27% |
| 151–300 | 640 | 97.66% |
| 301–500 | 342 | 98.25% |
| 501–750 | 653 | 97.86% |
| 750+ | 410 | 98.78% |



## Files

```text
10_Gelectra-large/
├── artifacts/
├── results/
│   ├── gelectra-large-confmat-3class.svg
│   ├── gelectra-large-confmat-human-ai.svg
│   └── gelectra-large-length-bin-summary.csv
├── Gelectra_large.ipynb
└── README.md
```

## Run

This stage requires the dataset prepared in [04 · HF 
dataset](../04_hf-dataset/).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run all cells in:

```text
Gelectra_large.ipynb
```

<div align="center">

[← 09 · XLM-RoBERTa-base](../09_xlm-roberta-base/) · [Project 
index](../README.md) · [11 · Luna6 baseline →](../11_luna6-baseline/)

</div>

