[← 04 · HF dataset](../04_hf-dataset/)

# 05 · TF-IDF + Logistic Regression

A classical **TF-IDF + Logistic Regression** baseline for the three-class 
classification task: **Human**, **AI-generated**, and **AI-assisted** 
text.

## Setup

The baseline uses a scikit-learn pipeline consisting of a TF-IDF 
vectorizer followed by Logistic Regression.

- TF-IDF: lowercase text, `max_df=0.98`, up to 200,000 features, sublinear 
term frequency
- Logistic Regression: `lbfgs` solver, maximum 1,000 iterations, random 
seed 17
- Hyperparameter search:
  - `min_df` ∈ {2, 5}
  - `ngram_range` ∈ {(1,1), (1,2)}
  - `C` ∈ {0.3, 1.0, 3.0}
- Model selection: `GridSearchCV`
- Cross-validation: 3-fold `StratifiedGroupKFold`
- Grouping: samples from the same source document remain in the same fold
- Selection metric: accuracy

The selected configuration was:

```text
C = 3.0
min_df = 5
ngram_range = (1, 1)
```

Because `C = 3.0` is the largest value included in the search grid, a 
wider hyperparameter search could potentially improve the baseline 
slightly.

No GPU is required.

## Test results

Evaluation on the held-out test set of **2,737 texts**:

| Accuracy | Macro-F1 | Human error | AI-generated error | AI-assisted error |
|---:|---:|---:|---:|---:|
| 91.23% | 91.22% | 8.53% | 3.40% | 14.38% |

AI-assisted text is the most difficult class for the baseline.

Performance is also strongly affected by text length. The pooled error 
rate is highest for texts of **70 words or fewer (23.06%)** and remains 
between **5.26% and 7.03%** for texts longer than 150 words.

Confusion matrices, length-based evaluation, and comparisons with the 
transformer models are available in the [main README](../README.md).

## Files

- `Logestic_Regression.ipynb` — training, model selection, evaluation, and 
text-length analysis
- `artifacts/tfidf-logreg-3class.joblib` — fitted TF-IDF + Logistic 
Regression pipeline
- `artifacts/tfidf-logreg-3class.json` — model metadata
- `results/` — confusion matrices and text-length analysis outputs

## Run

This stage requires the dataset prepared in [04 · HF 
dataset](../04_hf-dataset/).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run all cells in:

```text
Logestic_Regression.ipynb
```

<div align="center">

[← 04 · HF dataset](../04_hf-dataset/) · [Project index](../README.md) · 
[06 · mmBERT-small →](../06_mmBERT-small/)

</div>
