[← 11 · Luna6 Baseline](../11_luna6-baseline/)

# 12 · Learning Curves

This stage compares how the supervised models improve as the amount of 
available training data increases.

The same learning-curve procedure is applied separately to:

- TF-IDF + Logistic Regression
- mmBERT-small
- DistilBERT German
- mmBERT-base
- XLM-RoBERTa-base
- GeLECTRA-large

## Setup

Each model is trained on increasing fractions of the original training 
set:

- 1%
- 5%
- 12%
- 17.5%
- 25%
- 50%
- 100%

The corresponding training-set sizes are approximately:

| Training fraction | Samples |
|---:|---:|
| 1% | 250 |
| 5% | 1,249 |
| 12% | 2,998 |
| 17.5% | 4,373 |
| 25% | 6,246 |
| 50% | 12,493 |
| 100% | 24,986 |

A reproducible class-balanced training order is used so that every larger 
subset contains the samples from the smaller subsets.

For every training size, a fresh model is trained and the following values 
are recorded:

- training accuracy
- validation accuracy
- generalization gap
- training runtime

The validation set remains unchanged across all learning-curve 
experiments.

## Analysis

The learning curves show how quickly each model benefits from additional 
training data and whether performance has started to plateau.

Training and validation accuracy are plotted against the percentage of 
training data used.

For example, XLM-RoBERTa-base increases from approximately **46% 
validation accuracy at 1% of the training data** to approximately **91% at 
100%**, while the generalization gap becomes smaller as more data is 
added.

The same analysis is performed independently for all supervised models, 
allowing their data efficiency and generalization behavior to be compared 
directly.

## Files

```text
12_learning-curves/
├── Figures/
│   ├── distilbert-german-learning-curve.svg
│   ├── gelectra-large-learning-curve.svg
│   ├── mmbert-base-learning-curve.svg
│   ├── mmbert-small-learning-curve.svg
│   ├── tfidf-logreg-learning-curve.svg
│   └── xlm-roberta-base-learning-curve.svg
│
├── Notebooks/
│   ├── Gelectra_large_learning_curve.ipynb
│   ├── XLM_RoBERTa_base_learning_curve.ipynb
│   ├── distilbert_base_learning_curve.ipynb
│   ├── logreg-learning-curves.ipynb
│   ├── mmBERT_base_learning_curve.ipynb
│   └── mmBERT_small_learning_curve.ipynb
│
├── Results/
│   ├── distilbert-german-learning-curve.csv
│   ├── gelectra-large-learning-curve.csv
│   ├── mmbert-base-learning-curve.csv
│   ├── mmbert-small-learning-curve.csv
│   ├── tfidf-logreg-learning-curve.csv
│   └── xlm-roberta-base-learning-curve.csv
│
└── README.md
```

## Run

This stage requires the dataset prepared in [04 · HF 
dataset](../04_hf-dataset/).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run the corresponding notebook in:

```text
12_learning-curves/Notebooks/
```

Each notebook runs the same learning-curve experiment for its respective 
model and saves the resulting CSV file and learning-curve figure.

A CUDA-enabled GPU is recommended for the transformer models. The TF-IDF + 
Logistic Regression experiment can be run on CPU.

<div align="center">

[← 11 · Luna6 Baseline](../11_luna6-baseline/) · [Project 
index](../README.md) · [13 · Model Comparison →](../13_model-comparison/)

</div>
