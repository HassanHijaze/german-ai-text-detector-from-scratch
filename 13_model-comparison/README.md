[← 12 · Learning Curves](../12_learning-curves/)

# 13 · Model Comparison

This stage compares the learning curves and data efficiency of the six 
supervised classifiers developed in the project.

The comparison includes:

- TF-IDF + Logistic Regression
- DistilBERT German
- GeLECTRA-large
- mmBERT-base
- mmBERT-small
## Learning-curve comparison

Validation accuracy is compared using the same training-data fractions for all models:

| Model | 1% | 5% | 12% | 17.5% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|
| DistilBERT German | 46.42% | 69.10% | 75.05% | 78.87% | 82.35% | 87.82% | 90.50% |
| GeLECTRA-large | 69.34% | 83.34% | 90.09% | 93.74% | 93.67% | 93.98% | 95.84% |
| TF-IDF + Logistic Regression | 76.29% | 84.21% | 87.20% | 88.16% | 89.90% | 90.33% | 91.33% |
| XLM-RoBERTa-base | 51.72% | 60.91% | 78.49% | 81.11% | 84.62% | 88.58% | 91.60% |
| mmBERT-base | 65.97% | 87.20% | 91.43% | 93.43% | 94.22% | 95.11% | **96.25%** |
| mmBERT-small | 37.54% | 86.55% | 90.95% | 91.95% | 92.57% | 94.53% | 94.84% |

The classical TF-IDF baseline performs particularly well with very little training data, reaching **76.29% validation accuracy with only 1% of the training set**.

The transformer models benefit more strongly from additional data. At 100% 
training data, **mmBERT-base achieves the highest validation accuracy at 
96.25%**, followed by **GeLECTRA-large at 95.84%** and **mmBERT-small at 
94.84%**.

## Data efficiency

The comparison at 1%, 50%, and 100% training data highlights the different 
learning behavior of the models.

TF-IDF + Logistic Regression is the strongest model at the smallest 
training size, while mmBERT-base and GeLECTRA-large overtake it as 
substantially more training data becomes available.

This shows that the classical baseline is highly data-efficient, whereas 
the strongest transformer models require more training examples to reach 
their full performance.

## Files

```text
13_model-comparison/
├── results/
│   ├── data-efficiency-comparison-1-50-100.svg
│   ├── final-model-comparison.svg
│   ├── learning-curve-comparison-table.csv
│   └── learning-curve-model-comparison.svg
├── learning_curve_comparison.ipynb
└── README.md
```

## Run

This stage uses the learning-curve results generated in [12 · Learning 
Curves](../12_learning-curves/).

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Then run all cells in:

```text
learning_curve_comparison.ipynb
```

The notebook combines the individual learning-curve results and generates 
the comparison table, overall learning-curve figure, final-performance 
comparison, and data-efficiency analysis.

<div align="center">

[← 12 · Learning Curves](../12_learning-curves/) · [Project 
index](../README.md) · [14 · OOD Model Comparison 
→](../14_ood-model-comparison/)

</div>
