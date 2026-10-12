[← 07 · DistilBERT German](../07_distilbert-german/)

# 08 · mmBERT-base

Fine-tuning `jhu-clsp/mmBERT-base` for the three-class classification 
task: **Human**, **AI-generated**, and **AI-assisted** text.

## Setup

A fine-tuned transformer classifier based on `jhu-clsp/mmBERT-base`.

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and 
binary Human-vs-AI evaluation
- GPU recommended

## Test results

Evaluation on the held-out test set of **2,737 texts**:

| Accuracy | Macro-F1 | Human error | AI-generated error | AI-assisted 
error |
|---:|---:|---:|---:|---:|
| **96.64%** | **96.65%** | 5.58% | 0.99% | 3.51% |

### Class-specific errors

| Class | Error rate |
|---|---:|
| Human | 5.58% |
| AI-generated | 0.99% |
| AI-assisted | 3.51% |

The model performs particularly well on fully AI-generated text, with only 
**9 of 912 AI samples** misclassified.

Human-written text accounts for the largest number of errors, with **51 of 
914 Human samples** classified as AI-assisted.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.84% 
accuracy**.

The binary confusion matrix contains:

- **911** true positives
- **912** true negatives
- **2** false positives
- **1** false negative

### Length analysis

Performance improves considerably with increasing text length.

| Length | Samples | Accuracy |
|---|---:|---:|
| ≤70 | 399 | 89.47% |
| 71–150 | 293 | 96.59% |
| 151–300 | 640 | 96.88% |
| 301–500 | 342 | 99.42% |
| 501–750 | 653 | 98.47% |
| 750+ | 410 | 98.05% |


## Files

```text
08_mmbert-base/
├── artifacts/
├── results/
│   ├── mmbert-base-confmat-3class.svg
│   ├── mmbert-base-confmat-human-ai.svg
│   └── mmbert-base-length-bias-3class.csv
├── mmbert_base.ipynb
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
mmbert_base.ipynb
```

<div align="center">

[← 07 · DistilBERT German](../07_distilbert-german/) · [Project 
index](../README.md) · [09 · XLM-RoBERTa-base →](../09_xlm-roberta-base/)

</div>
