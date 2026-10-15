[← 05 · Luna 6 Zero-Shot Baseline](../05_luna-6-baseline/)

# 06 · TF-IDF + Logistic Regression Baseline

A classical machine-learning baseline for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model uses TF-IDF word and bigram features with Logistic Regression.

## Tuning

Hyperparameters were selected using **3-fold StratifiedGroupKFold cross-validation** on the training data. The held-out test set was used only for the final evaluation.

| Parameter | Values tested | Best |
|---|---|---:|
| `C` | 0.3, 1.0, 3.0 | **3.0** |
| `min_df` | 2, 5 | **5** |
| `ngram_range` | `(1,1)`, `(1,2)` | **(1,2)** |
| Maximum features | 200,000 | **200,000** |


## Results


| **Split**  | **Accuracy** |
| ---------- | ------------ |
| Train      | 99.55% |
| Validation | 91.47% |
| **Test**   | **91.23%** (macro-F1 **91.22%**) |


| **True class** | **Correct** | **Error** | **Most common mistake** |
|---|---:|---:|---|
| Human | 91.5% | 8.5% | called AI-assisted (67 of 914) |
| AI-generated | 96.6% | 3.4% | called AI-assisted (26 of 912) |
| AI-assisted | 85.6% | 14.4% | called Human (112 of 911) |

The model performs best on **AI-generated text** and struggles most with **AI-assisted text**, particularly when AI-assisted samples are classified as Human.


## Text Length

Performance depends strongly on the amount of text available, especially for very short samples.

| Words | Samples | Accuracy | Human error | AI-generated error | AI-assisted error |
|---|---:|---:|---:|---:|---:|
| ≤70 | 399 | 76.94% | 19.40% | 11.19% | 38.93% |
| 71–150 | 293 | 91.81% | 9.57% | 2.46% | 15.58% |
| 151–300 | 640 | 92.97% | 7.73% | 2.86% | 11.70% |
| 301–500 | 342 | **94.74%** | 4.72% | 3.37% | 6.80% |
| 501–750 | 653 | 94.18% | 5.58% | 1.57% | 9.61% |
| 750+ | 410 | 94.39% | 6.43% | **0.00%** | 10.07% |

The largest performance drop occurs for texts of **70 words or fewer**, where AI-assisted samples are particularly difficult. From around 150 words onward, accuracy remains relatively stable at approximately **93–95%**.
## How to Run

The trained three-class model is available on [**Hugging Face**](https://huggingface.co/HassanHijazi/german-ai-text-tfidf-logreg) and can be downloaded with `huggingface_hub` and loaded using `joblib`.

To reproduce the training and evaluation locally, place the dataset at `data/hf-dataset-3class-full`, start JupyterLab from the repository root with `jupyter lab`, and run `06_logreg-baseline/logestic_regression.ipynb` from top to bottom.

No API key or GPU is required.

---

[← 05 · Luna 6 Zero-Shot Baseline](../05_luna-6-baseline/) · [07 · mmBERT-small →](../07_mmbert-small/)