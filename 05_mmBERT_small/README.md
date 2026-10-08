# mmBERT-small

Fine-tuning 
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) 
for three-class German AI-text detection.

The classifier distinguishes between:

- Human-written text
- AI-generated text
- AI-assisted text

## Results

The final model reached **95.98% test accuracy** and **95.63% validation 
accuracy**.

| Split | Accuracy | Brier score | Log loss |
|---|---:|---:|---:|
| Train | 99.83% | 0.0060 | 0.0336 |
| Validation | 95.63% | 0.0714 | 0.1464 |
| Test | 95.98% | 0.0689 | 0.1397 |

Temperature scaling improved validation calibration:

| Model | Brier score | Log loss |
|---|---:|---:|
| Uncalibrated | 0.0825 | 0.3858 |
| Temperature-scaled | 0.0714 | 0.1464 |

The learned calibration temperature was **3.9188**.

## Error profile

On the three-class test set, most errors occur between **Human** and 
**AI-assisted** text, while fully AI-generated samples are classified 
particularly reliably.

The additional binary Human-vs-AI evaluation produced:

- **911** true positives
- **909** true negatives
- **5** false positives
- **1** false negative

## Length analysis

Performance improves noticeably with text length. Accuracy rises from 
about **85.2% for texts up to 70 words** to around **98% for samples above 
300 words**, indicating that very short texts remain the most difficult 
cases.

## Learning curve

Increasing the amount of training data consistently improves validation 
performance.

| Training fraction | Samples | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|
| 1% | 249 | 94.38% | 73.68% |
| 10% | 2,498 | 99.88% | 92.40% |
| 12% | 2,998 | 99.83% | 93.05% |
| 25% | 6,246 | 99.87% | 93.84% |
| 50% | 12,493 | 99.29% | 95.32% |

## Artifacts

The trained model, tokenizer, and detector configuration are stored in:

```text
artifacts/mmbert-small-3class/
