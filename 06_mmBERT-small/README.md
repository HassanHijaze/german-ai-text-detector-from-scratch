[← 05 · TF-IDF + Logistic Regression](../05_logreg-baseline/)

# 06 · mmBERT-small

Fine-tuning [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) for three-class German AI-text detection.

The classifier distinguishes between:

- Human-written text
- AI-generated text
- AI-assisted text

## Setup

A fine-tuned transformer classifier based on `jhu-clsp/mmBERT-small`.

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Calibration metrics: Brier score and log loss
- Additional analyses: class-specific errors, text-length performance, binary Human-vs-AI evaluation, and learning curve

## Test results

Evaluation on the held-out test set of **2,737 texts**.

The final model reached **95.94% test accuracy** and **95.70% validation accuracy**.

| Split | Accuracy | Brier score | Log loss |
|---|---:|---:|---:|
| Train | 99.89% | 0.0062 | 0.0331 |
| Validation | 95.70% | 0.0701 | 0.1435 |
| Test | 95.94% | 0.0660 | 0.1369 |

### Class-specific errors

| Class | Error rate |
|---|---:|
| Human | 6.24% |
| AI-generated | 0.99% |
| AI-assisted | 4.94% |

The model is particularly reliable at identifying fully AI-generated text, with only **9 of 912 AI samples** misclassified.

Most remaining errors occur between **Human** and **AI-assisted** text.

The test confusion matrix is:

```text
[[857   0  57]
 [  0 903   9]
 [ 40   5 866]]
```

## Length analysis

Performance varies with text length, with very short texts remaining the most difficult samples.

The analysis uses the following word-count bins:

- ≤70
- 71–150
- 151–300
- 301–500
- 501–750
- 750+

The full length-based evaluation is stored in:

```text
results/mmbert-small-length-bias-3class.csv
```

## Learning curve

Increasing the amount of training data consistently improves validation performance.

| Training fraction | Samples | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|
| 1% | 249 | 94.38% | 73.68% |
| 10% | 2,498 | 99.88% | 92.40% |
| 12% | 2,998 | 99.83% | 93.05% |
| 25% | 6,246 | 99.87% | 93.84% |
| 50% | 12,493 | 99.29% | 95.32% |

The largest gains occur at smaller training-set sizes, while improvements become more gradual as additional data is added.

## Files

- model notebook: training, calibration, evaluation, confusion matrices, and length analysis
- `artifacts/mmbert-small-3class/`: trained model, tokenizer, model configuration, and detector metadata
- `results/mmbert-small-confmat-3class.svg`: three-class confusion matrix
- `results/mmbert-small-confmat-human-ai.svg`: Human-vs-AI confusion matrix
- `results/mmbert-small-length-bias-3class.csv`: text-length analysis

## Run

This stage requires the dataset prepared in [04 · HF dataset](../04_hf-dataset/).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run all cells of the mmBERT-small notebook.

<div align="center">

[← 05 · TF-IDF + Logistic Regression](../05_logreg-baseline/) · [Project index](../README.md) · [07 · DistilBERT →](../07_distilbert-base-german-cased/)

</div>