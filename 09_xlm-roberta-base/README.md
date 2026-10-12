[← 08 · mmBERT-base](../08_mmbert-base/)

# 09 · XLM-RoBERTa-base

Fine-tuning `xlm-roberta-base` for the three-class classification task: 
**Human**, **AI-generated**, and **AI-assisted** text.

## Setup

A fine-tuned XLM-RoBERTa transformer classifier.

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
| **91.09%** | **91.03%** | 21.66% | 0.88% | 4.17% |

### Class-specific errors

| Class | Error rate |
|---|---:|
| Human | 21.66% |
| AI-generated | 0.88% |
| AI-assisted | 4.17% |

The model performs particularly well on fully AI-generated text, with only 
**8 of 912 AI samples** misclassified.

Human-written text is the most difficult class, with **198 of 914 Human 
samples** misclassified, primarily as AI-assisted text.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.56% 
accuracy**.

The binary confusion matrix contains:

- **912** true positives
- **906** true negatives
- **8** false positives
- **0** false negatives

### Length analysis

Performance varies with text length, with very short texts being the most 
difficult.

| Length | Samples | Accuracy |
|---|---:|---:|
| ≤70 | 399 | 82.96% |
| 71–150 | 293 | 91.13% |
| 151–300 | 640 | 91.72% |
| 301–500 | 342 | 94.44% |
| 501–750 | 653 | 93.57% |
| 750+ | 410 | 91.22% |



## Files

```text
09_xlm-roberta-base/
├── artifacts/
├── results/
│   ├── xlm-roberta-base-confmat-3class.svg
│   ├── xlm-roberta-base-confmat-human-ai.svg
│   └── xlm-roberta-base-length-bias-3class.csv
├── xlm_roberta_base.ipynb
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
xlm_roberta_base.ipynb
```

<div align="center">

[← 08 · mmBERT-base](../08_mmbert-base/) · [Project index](../README.md) 
· [10 · GeLECTRA-large →](../10_Gelectra-large/)

</div>
