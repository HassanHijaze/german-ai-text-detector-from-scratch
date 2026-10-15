[← 11 · GeLECTRA-large](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/11_gelectra-large)

# 12 · Learning Curves

This stage evaluates how the six supervised models improve as the amount of available training data increases.

Each model is trained on progressively larger subsets of the original training set and evaluated on the same validation set.

The experiment includes:

- TF-IDF + Logistic Regression
- mmBERT-small
- DistilBERT German
- mmBERT-base
- XLM-RoBERTa base
- GeLECTRA-large

## Setup

The training subsets range from approximately **1% of the training data (250 texts)** to the complete **24,986-text training set**.

The subsets are reproducible and class-balanced, and the validation set remains unchanged across all runs.

For each training size, the following values are recorded:

- Training accuracy
- Validation accuracy
- Generalization gap
- Training runtime

The held-out test set is not used during the learning-curve experiments.

## Validation Accuracy

The table below contains the validation accuracies from the individual learning-curve result files.

| **Model** | **1%** | **5%** | **12%** | **17.5%** | **25%** | **50%** | **100%** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DistilBERT German | 46.4% | 69.1% | 75.1% | 78.9% | 82.3% | 87.8% | 90.5% |
| GeLECTRA-large | 69.3% | 83.3% | 90.1% | 93.7% | 93.7% | 94.0% | 95.8% |
| TF-IDF + Logistic Regression | **76.3%** | 84.2% | 87.2% | 88.2% | 89.0% | 90.3% | 91.3% |
| XLM-RoBERTa base | 51.7% | 60.9% | 78.5% | 81.1% | 84.6% | 88.6% | 91.6% |
| mmBERT-base | 66.0% | **87.2%** | **91.4%** | **93.4%** | **94.2%** | **95.1%** | **96.2%** |
| mmBERT-small | 37.5% | 86.5% | 90.9% | 91.9% | 92.6% | 94.5% | 94.8% |


## Results

With only **250 training texts**, TF-IDF + Logistic Regression performs best with **76.3% validation accuracy**. At this training size, the transformer models range from **37.5% to 69.3%**.

The picture changes quickly as more training data becomes available. At **1,249 texts**, mmBERT-base reaches **87.2%** and mmBERT-small **86.5%**, both already exceeding TF-IDF at the same training size.

At approximately **3,000 texts**, mmBERT-base reaches **91.4% validation accuracy**, which is already slightly higher than the **91.3%** reached by TF-IDF using the complete training set.

GeLECTRA-large also improves quickly, reaching **90.1% with 2,499 texts** and **93.7% with 4,373 texts**.

With the complete training set, **mmBERT-base achieves the highest validation accuracy at 96.2%**, followed by **GeLECTRA-large at 95.8%** and **mmBERT-small at 94.8%**.

## Data Efficiency

The learning curves show different behavior across the models.

**TF-IDF + Logistic Regression** is particularly effective when very little training data is available, but its improvements become smaller as the training set grows.

The **mmBERT models** benefit strongly from the first increases in training size. mmBERT-base rises from **66.0% at 1%** to **91.4% at 12%**, while mmBERT-small rises from **37.5% to 90.9%** over the same range.

**GeLECTRA-large** also learns quickly, increasing from **69.3% at 1%** to **90.1% at 12%**.

**DistilBERT German** and **XLM-RoBERTa base** improve more gradually and continue to gain substantially as more training data is added.

## Performance from 50% to 100%

The improvement between half of the training data and the complete training set shows how strongly each model still benefits from additional data.

| **Model** | **50%** | **100%** | **Gain** |
| --- | ---: | ---: | ---: |
| XLM-RoBERTa base | 88.6% | 91.6% | **+3.0 pp** |
| DistilBERT German | 87.8% | 90.5% | **+2.7 pp** |
| GeLECTRA-large | 94.0% | 95.8% | **+1.9 pp** |
| mmBERT-base | 95.1% | 96.2% | **+1.1 pp** |
| TF-IDF + Logistic Regression | 90.3% | 91.3% | **+1.0 pp** |
| mmBERT-small | 94.5% | 94.8% | **+0.3 pp** |

Among the strongest models, mmBERT-small shows the smallest additional gain from 50% to 100%, while mmBERT-base and GeLECTRA-large continue to improve.

XLM-RoBERTa base and DistilBERT German show the largest gains over the same interval.


## How to Run

This stage requires the dataset prepared in [04 · HF Dataset](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/04_hf-dataset).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run the corresponding notebook in:

```text
12_learning-curves/Notebooks/
```

Each notebook trains one model on the defined training subsets and saves the resulting learning-curve CSV file and figure.

A CUDA-enabled GPU is recommended for the transformer models. TF-IDF + Logistic Regression can be run on CPU.

---

[← 11 · GeLECTRA-large](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/11_gelectra-large) · [13 · Model Comparison →](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/13_model-comparison)