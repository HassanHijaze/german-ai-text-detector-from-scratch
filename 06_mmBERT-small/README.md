# mmBERT-small

Fine-tuning 
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) 
for three-class German AI-text detection.

The classifier distinguishes between:

- Human-written text
- AI-generated text
- AI-assisted text

## Results

The final model reached **95.94% test accuracy** and **95.70% validation accuracy**.

| Split | Accuracy | Brier score | Log loss |
|---|---:|---:|---:|
| Train | 99.89% | 0.0062 | 0.0331 |
| Validation | 95.70% | 0.0701 | 0.1435 |
| Test | 95.94% | 0.0660 | 0.1369 |


The test confusion matrix was:

```text
[[857   0  57]
 [  0 903   9]
 [ 40   5 866]]
```

This corresponds to the following class-specific error rates:

| Class | Error rate |
|---|---:|
| Human | 6.24% |
| AI | 0.99% |
| AI-assisted | 4.94% |

The model is especially strong at recognizing fully AI-generated text, with only **9 of 912 AI samples** misclassified. The largest remaining source of error is the distinction between Human and AI-assisted text.

## Length analysis

Performance improves noticeably with text length. Very short texts remain the most difficult cases, while longer samples are classified more reliably.

The analysis uses the following word-count bins:

- ≤70
- 71–150
- 151–300
- 301–500
- 501–750
- 750+

The corresponding length-bias results are stored in:

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

The learning-curve results show that mmBERT-small benefits substantially from additional training data, particularly at smaller dataset sizes. Performance gains become more gradual once a larger fraction of the training set is used.

## Artifacts

The trained model, tokenizer, and detector configuration are stored in:

```text
artifacts/mmbert-small-3class/
```

The artifact directory contains the files required for later inference, including the trained model weights, tokenizer, model configuration, and detector metadata.

## Result files

The main evaluation outputs include:

```text
results/mmbert-small-confmat-3class.svg
results/mmbert-small-confmat-human-ai.svg
results/mmbert-small-length-bias-3class.csv
```

The three-class confusion matrix summarizes performance across Human, AI, and AI-assisted samples, while the Human-vs-AI matrix provides an additional binary evaluation focused on distinguishing Human from AI-generated text.