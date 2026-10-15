[← 09 · mmBERT-base](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/09_mmbert-base)

# 10 · XLM-RoBERTa base

A fine-tuned transformer model for classifying German text as **Human**, **AI-generated**, or **AI-assisted**.

The model is based on [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) and uses a maximum sequence length of 512 tokens.

## Setup

A fine-tuned XLM-RoBERTa transformer classifier for German text.

- Task: three-class sequence classification
- Classes: Human, AI-generated, AI-assisted
- Maximum sequence length: 512 tokens
- Evaluation: train, validation, and held-out test splits
- Additional analyses: class-specific errors, text-length performance, and binary Human-vs-AI evaluation
- GPU recommended

## Results

| **Split** | **Accuracy** |
| --------- | ------------ |
| **Test**  | **91.09%** (macro-F1 **91.03%**) |

| **True class** | **Correct** | **Error** |
| -------------- | ----------- | --------- |
| Human          | 78.34%      | 21.66%    |
| AI-generated   | **99.12%**  | **0.88%** |
| AI-assisted    | 95.83%      | 4.17%     |

The model performs particularly well on **AI-generated text**, with only **8 of 912 samples** misclassified.

Human-written text is the most difficult class, with **198 of 914 Human samples** misclassified, primarily as AI-assisted text.

### Human vs AI

For the binary Human-vs-AI evaluation, the model reached **99.56% accuracy**.

The binary confusion matrix contains:

- **912** true positives
- **906** true negatives
- **8** false positives
- **0** false negatives

## Text Length

Performance varies with text length, with very short texts being the most difficult.

| **Words** | **Samples** | **Accuracy** |
| --------- | ----------- | ------------ |
| ≤70       | 399         | 82.96%       |
| 71–150    | 293         | 91.13%       |
| 151–300   | 640         | 91.72%       |
| 301–500   | 342         | **94.44%**   |
| 501–750   | 653         | 93.57%       |
| 750+      | 410         | 91.22%       |

The largest performance drop occurs for texts of **70 words or fewer**. Performance improves substantially for longer texts, reaching its highest accuracy for samples between **301 and 500 words**.

## How to Run

The trained three-class model is available on [**Hugging Face**](https://huggingface.co/HassanHijazi/german-ai-text-xlm-roberta-base) and can be loaded directly with `transformers`.

To reproduce the training and evaluation locally, place the dataset at `data/hf-dataset-3class-full`, start JupyterLab from the repository root with `jupyter lab`, and run `10_xlm-roberta-base/xlm_roberta_base.ipynb` from top to bottom.

A GPU is recommended for training and evaluation.

---

[← 09 · mmBERT-base](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/09_mmbert-base) · [11 · GeLECTRA-large →](https://github.com/HassanHijaze/german-ai-text-detector-from-scratch/blob/main/11_gelectra-large)