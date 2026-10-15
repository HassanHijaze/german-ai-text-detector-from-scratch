[← 06 · TF-IDF + Logistic Regression](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/06_logreg-baseline)

# 07 · mmBERT-small

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) and uses a maximum sequence length of 512 tokens.

## Results

| **Split**  | **Accuracy**                     |
| ---------- | -------------------------------- |
| Train      | 99.89%                           |
| Validation | 95.70%                           |
| **Test**   | **95.94%** (macro-F1 **95.95%**) |

| **True class** | **Correct** | **Error** |
| -------------- | ----------- | --------- |
| Human          | 93.76%      | 6.24%     |
| AI-generated   | **99.01%**  | **0.99%** |
| AI-assisted    | 95.06%      | 4.94%     |

The model performs particularly well on **AI-generated text**, with only **9 of 912 samples** misclassified.

Most remaining errors occur between **Human** and **AI-assisted** text.

## Text Length

Performance improves substantially as more text becomes available.

| **Words** | **Samples** | **Accuracy** |
| --------- | ----------- | ------------ |
| ≤70       | 399         | 87.97%       |
| 71–150    | 293         | 95.56%       |
| 151–300   | 640         | 95.63%       |
| 301–500   | 342         | 98.83%       |
| 501–750   | 653         | 98.01%       |
| 750+      | 410         | **98.78%**   |

The largest performance drop occurs for texts of **70 words or fewer**. From around 150 words onward, accuracy remains consistently high at approximately **96–99%**.

## How to Run

The trained three-class model is available on [**Hugging Face**](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-small) and can be loaded directly with `transformers`.

To reproduce the training and evaluation locally, place the dataset at `data/hf-dataset-3class-full`, start JupyterLab from the repository root with `jupyter lab`, and run `07_mmbert-small/mmBERT_small.ipynb` from top to bottom.

A GPU is recommended for training and evaluation.

---

[← 06 · TF-IDF + Logistic Regression](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/06_logreg-baseline) · [08 · DistilBERT German →](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/08_distilbert-german)