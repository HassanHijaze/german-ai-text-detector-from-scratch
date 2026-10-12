[← 13 · Model Comparison](../13_model-comparison/)

# 14 · OOD Model Comparison

This stage evaluates the trained models on a separate out-of-distribution (OOD) dataset that was not used during training or the original held-out evaluation.

## OOD Data Construction

The external `Data-test` was created using the same general data collection and generation pipeline as the main project, but with different source material to provide a stronger out-of-distribution evaluation.

### Human texts

The **290 Human samples** were collected using the same workflow as:

[01 · Human Data](../01_human-data/)

and the notebook:

```text
collect_human_data.ipynb
```

For the OOD dataset, different Human sources were used:

- **German Wikinews** — news
- **German Wikipedia** — encyclopedic
- **German Wikivoyage** — travel
- **German Wikibooks** — instructional

These sources differ from the Human sources used for the original training dataset and introduce new writing domains for evaluation.

### AI-generated and AI-assisted texts

The AI-generated and AI-assisted samples were created using the same generation procedure as:

[02 · AI-Generated Data](../02_ai-generated-data/)

and the notebook:

```text
generate_ai_data.ipynb
```

The Human texts were processed using the same four model providers used for the main dataset generation:

- GPT-5.6 Luna
- Gemini 3.8 Flash
- DeepSeek Flash
- GPT-6 Luna

The final external `Data-test` contains **869 texts**:

- **290 Human**
- **289 AI-generated**
- **290 AI-assisted**

The trained classifiers are then evaluated on this dataset to measure how well they generalize to new sources and domains.

## Models

Six models are evaluated:

- TF-IDF + Logistic Regression
- DistilBERT
- GeLECTRA-large
- mmBERT-base
- mmBERT-small
- XLM-RoBERTa-base

## Results

| Model | 3-Class Accuracy | Macro F1 | Human vs AI Accuracy |
|---|---:|---:|---:|
| mmBERT-base | **96.55%** | **0.9658** | **98.27%** |
| GeLECTRA-large | 96.32% | 0.9633 | 96.37% |
| XLM-RoBERTa-base | 91.83% | 0.9192 | 91.36% |
| DistilBERT | 89.41% | 0.8953 | 90.33% |
| mmBERT-small | 81.13% | 0.8021 | 74.44% |
| TF-IDF + Logistic Regression | 79.40% | 0.7921 | 96.89% |

`mmBERT-base` achieved the strongest overall OOD performance, closely followed by `GeLECTRA-large`.




The evaluation also includes:

- class-specific error rates
- 3-class confusion matrices
- Human-vs-AI confusion matrices
- model agreement analysis
- a shared 20-sample prediction comparison
- class probabilities for the same 20 samples

## Requirements

Install the required dependencies from the project root:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for evaluating the transformer models.

## Notebook

Run the analysis from the project directory after downloading the model artifacts:

```text
ood_model_comparison.ipynb
```

The notebook contains the complete OOD evaluation and automatically generates the comparison tables, prediction probabilities, agreement analysis, and confusion matrices.

## Files

```text
14_ood-model-comparison/
├── 2-class-matrices/
├── 3-class-matrices/
├── tables/
├── README.md
└── ood_model_comparison.ipynb
```