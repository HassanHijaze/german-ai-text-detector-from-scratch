[← 08 · DistilBERT German](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/08_distilbert-german)

# 09 · mmBERT-base

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) and uses a maximum sequence length of 512 tokens.

## Setup

A fine-tuned mmBERT-base transformer classifier for German text.

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and binary Human-vs-AI evaluation
- GPU recommended

## Results

| **Split** | **Accuracy** |
| --------- | ------------ |
| **Test**  | **96.64%** (macro-F1 **96.65%**) |

| **True class** | **Correct** | **Error** |
| -------------- | ----------- | --------- |
| Human          | 94.42%      | 5.58%     |
| AI-generated   | **99.01%**  | **0.99%** |
| AI-assisted    | 96.49%      | 3.51%     |

The model performs particularly well on **AI-generated text**, with only **9 of 912 samples** misclassified.

Human-written text accounts for the largest number of errors, with **51 of 914 Human samples** classified as AI-assisted.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.84% accuracy**.

The binary confusion matrix contains:

- **911** true positives
- **912** true negatives
- **2** false positives
- **1** false negative

## Text Length

Performance improves considerably as more text becomes available.

| **Words** | **Samples** | **Accuracy** |
| --------- | ----------- | ------------ |
| ≤70       | 399         | 89.47%       |
| 71–150    | 293         | 96.59%       |
| 151–300   | 640         | 96.88%       |
| 301–500   | 342         | **99.42%**   |
| 501–750   | 653         | 98.47%       |
| 750+      | 410         | 98.05%       |

The largest performance drop occurs for texts of **70 words or fewer**. From around 150 words onward, accuracy remains consistently high at approximately **97–99%**.

## How to Run

The trained three-class model is available on [**Hugging Face**](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-base) and can be loaded directly with `transformers`.

To reproduce the training and evaluation locally, place the dataset at `data/hf-dataset-3class-full`, start JupyterLab from the repository root with `jupyter lab`, and run `09_mmbert-base/mmbert_base.ipynb` from top to bottom.

A GPU is recommended for training and evaluation.

---

[← 08 · DistilBERT German](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/08_distilbert-german) · [10 · XLM-RoBERTa base →](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/10_xlm-roberta-base)