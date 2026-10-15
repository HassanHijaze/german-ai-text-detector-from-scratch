[← 06 · mmBERT-small](../06_mmBERT-small/)

# 07 · DistilBERT German

Fine-tuning a German DistilBERT model for the three-class classification 
task: **Human**, **AI-generated**, and **AI-assisted** text.

## Setup

A fine-tuned DistilBERT transformer classifier for German text.

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
| **90.83%** | **90.82%** | 19.91% | 2.08% | 5.49% |


### Class-specific errors

| Class | Error rate |
|---|---:|
| Human | 19.91% |
| AI-generated | 2.08% |
| AI-assisted | 5.49% |

The model performs particularly well on fully AI-generated text, with only 
**19 of 912 AI samples** misclassified.

Human-written text is the most difficult class, with **182 of 914 Human 
samples** classified as either AI-generated or AI-assisted.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.56% 
accuracy**.

The binary confusion matrix contains:

- **910** true positives
- **908** true negatives
- **6** false positives
- **2** false negatives

### Length analysis

Performance varies with text length, with very short texts being the most 
difficult.

| Length | Samples | Accuracy |
|---|---:|---:|
| ≤70 | 399 | 83.21% |
| 71–150 | 293 | 90.10% |
| 151–300 | 640 | 92.19% |
| 301–500 | 342 | 93.86% |
| 501–750 | 653 | 92.50% |
| 750+ | 410 | 91.46% |


## Files

```text
07_distilbert-german/
├── artifacts/
├── results/
│   ├── distilbert-confmat-3class.svg
│   ├── distilbert-confmat-human-ai.svg
│   └── distilbert-length-bias-3class.csv
├── distilbert_base_german_cased.ipynb
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
distilbert_base_german_cased.ipynb
```

<div align="center">

[← 06 · mmBERT-small](../06_mmBERT-small/) · [Project 
index](../README.md) · [08 · mmBERT-base →](../08_mmbert-base/)

</div>
