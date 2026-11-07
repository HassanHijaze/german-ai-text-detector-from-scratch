[← 06 · TF-IDF baseline](../06_logreg-baseline/) · [Project index](../README.md) · [12 · Learning Curves →](../12_learning-curves/)

# Fine-tuning

One set of scripts to fine-tune and evaluate all five transformer detectors with the same settings.

## Results

Held-out test set (2,737 texts).

| Model | Test accuracy | Macro-F1 | Human error | AI-generated error | AI-assisted error | Hugging Face |
|---|---:|---:|---:|---:|---:|---|
| [GeLECTRA-large](./results/gelectra-large/) | **96.89%** | **96.91%** | **4.70%** | 1.54% | **3.07%** | [model](https://huggingface.co/HassanHijazi/german-ai-text-gelectra-large) |
| [mmBERT-base](./results/mmbert-base/) | 96.64% | 96.65% | 5.58% | 0.99% | 3.51% | [model](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-base) |
| [mmBERT-small](./results/mmbert-small/) | 95.94% | 95.95% | 6.24% | 0.99% | 4.94% | [model](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-small) |
| [XLM-RoBERTa base](./results/xlm-roberta-base/) | 91.09% | 91.03% | 21.66% | **0.88%** | 4.17% | [model](https://huggingface.co/HassanHijazi/german-ai-text-xlm-roberta-base) |
| [DistilBERT German](./results/distilbert-german/) | 90.83% | 90.82% | 19.91% | 2.08% | 5.49% | [model](https://huggingface.co/HassanHijazi/german-ai-text-distilbert-german) |

Each model's folder has its confusion matrices and length-bias table.

## Setup

All models use the same settings ([`config.yaml`](./config.yaml)): learning rate 2e-5, batch size 8, 3 epochs, weight decay 0.01, 235 warmup steps, early stopping on validation accuracy, max 512 tokens, seed 17. Probabilities are calibrated with temperature scaling on the validation set.

## Run

Place the dataset at `data/hf-dataset-3class-full` in the repository root (it is private), then:

```bash
pip install -r requirements.txt
cd 07_fine-tuning

python scripts/run.py mmbert-base                   # train + evaluate one model
python scripts/run.py all                           # all five models
python scripts/run.py mmbert-base --evaluate-only   # evaluate the published model, no training
```

New evaluation results are written to `results/<name>/new-run/`, so the reported results in `results/<name>/` are never overwritten. Trained models go to `artifacts/` (not in Git). A GPU is recommended for training.

## Files

```text
07_fine-tuning/
├── config.yaml        shared settings and model list
├── scripts/
│   ├── common.py      config, paths, dataset loading, model-name check
│   ├── train.py       fine-tune one model
│   ├── evaluate.py    evaluate one model (local or from Hugging Face)
│   └── run.py         train + evaluate one model or all five
├── results/<name>/    reported results per model; new evaluations go to new-run/
├── archive/           the original notebooks
└── README.md
```

<div align="center">

[← 06 · TF-IDF baseline](../06_logreg-baseline/) · [Project index](../README.md) · [12 · Learning Curves →](../12_learning-curves/)

</div>