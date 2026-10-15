# German AI Text Detector

A three-class machine learning project for distinguishing **human-written**, **AI-generated**, and **AI-assisted** German text.

Unlike conventional binary Human-vs-AI detectors, this project treats **AI-assisted writing** — human text rewritten or polished by a language model — as a separate class.

**Models on Hugging Face:** [GeLECTRA-large](https://huggingface.co/HassanHijazi/german-ai-text-gelectra-large) · [mmBERT-base](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-base) · [mmBERT-small](https://huggingface.co/HassanHijazi/german-ai-text-mmbert-small) · [TF-IDF + Logistic Regression](https://huggingface.co/HassanHijazi/german-ai-text-tfidf-logreg) · [XLM-RoBERTa base](https://huggingface.co/HassanHijazi/german-ai-text-xlm-roberta-base) · [DistilBERT German](https://huggingface.co/HassanHijazi/german-ai-text-distilbert-german)

## At a glance

- **30,629 texts** across three classes
- **6 supervised models** + **1 zero-shot LLM baseline**
- Top test accuracy: **96.89% — GeLECTRA-large**
- Top OOD accuracy: **96.55% — mmBERT-base**
- Evaluation includes learning curves, class errors, length bias, calibration, and OOD generalization


## Repository guide

| Stage | Folder | Purpose |
|---:|---|---|
| 01 | [`01_human-data`](./01_human-data/) | Collect and clean German academic human text |
| 02 | [`02_ai-generated-data`](./02_ai-generated-data/) | Generate AI-generated and AI-assisted texts |
| 03 | [`03_pangram-test`](./03_pangram-test/) | External Human-vs-AI sanity check |
| 04 | [`04_hf-dataset`](./04_hf-dataset/) | Build the final Hugging Face dataset |
| 05 | [`05_luna-6-baseline`](./05_luna-6-baseline/) | Evaluate Luna 6 as a zero-shot LLM baseline |
| 06 | [`06_logreg-baseline`](./06_logreg-baseline/) | Train and evaluate the TF-IDF + Logistic Regression baseline |
| 07 | [`07_mmbert-small`](./07_mmbert-small/) | Fine-tune and evaluate mmBERT-small |
| 08 | [`08_distilbert-german`](./08_distilbert-german/) | Fine-tune and evaluate DistilBERT German |
| 09 | [`09_mmbert-base`](./09_mmbert-base/) | Fine-tune and evaluate mmBERT-base |
| 10 | [`10_xlm-roberta-base`](./10_xlm-roberta-base/) | Fine-tune and evaluate XLM-RoBERTa base |
| 11 | [`11_gelectra-large`](./11_gelectra-large/) | Fine-tune and evaluate GeLECTRA-large |
| 12 | [`12_learning-curves`](./12_learning-curves/) | Analyze model performance across different training-set sizes |
| 13 | [`13_model-comparison`](./13_model-comparison/) | Compare all supervised models on the held-out test set |
| 14 | [`14_ood-model-comparison`](./14_ood-model-comparison/) | Evaluate model generalization on out-of-distribution data |

## Dataset

The dataset contains **30,629 German texts** in three classes:

| Class | Samples |
|---|---:|
| Human | 10,227 |
| AI-generated | 10,203 |
| AI-assisted | 10,199 |
| **Total** | **30,629** |

Human texts were collected from **SSOAR, peDOCS, and Refubium** and consist of openly licensed German academic documents published before the widespread use of generative AI.

For most human samples, corresponding AI texts were created using **GPT-5.6 Luna, Gemini 3.8 Flash, DeepSeek V4 Flash, and GPT-6 Luna**:

- **AI-generated:** a new text on the same topic with independent wording and structure.
- **AI-assisted:** the original human text is rewritten and polished while preserving its meaning and structure.

The dataset was split at document level to reduce source leakage:

| Split | Samples |
|---|---:|
| Train | 24,986 |
| Validation | 2,906 |
| Test | 2,737 |

### OOD dataset

Generalization was evaluated on an additional **869-text out-of-distribution dataset** containing:

- 290 Human
- 289 AI-generated
- 290 AI-assisted

Human texts were drawn from **German Wikipedia, Wikinews, Wikivoyage, and Wikibooks**, introducing encyclopedic, news, travel, and instructional domains not present in the academic training data.


## Models

Six supervised classifiers were trained or fine-tuned on the same three-class dataset. In addition, Luna-6 was evaluated as a zero-shot LLM baseline without task-specific training.

| Model | Type | Parameters |
|---|---|---:|
| TF-IDF + Logistic Regression | Classical baseline | ≤0.60M coefficients* |
| [DistilBERT German](https://huggingface.co/distilbert/distilbert-base-german-cased) | Transformer | ~67M |
| [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | Transformer | 140M |
| [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base) | Transformer | ~278M |
| [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) | Transformer | 307M |
| [GeLECTRA-large](https://huggingface.co/deepset/gelectra-large) | Transformer | ~335M |
| Luna 6 (OpenAI API) | Zero-shot LLM baseline | Not publicly disclosed |




## Results

## Results

All six supervised models and the Luna 6 zero-shot baseline were evaluated on the same held-out test set of **2,737 texts**. The supervised models were additionally evaluated on a separate out-of-distribution (OOD) dataset of **869 texts** from previously unseen source domains.

| Model | Test Accuracy | Macro-F1 | OOD Accuracy | Test Human false-positive rate |
|---|---:|---:|---:|---:|
| GeLECTRA-large | **96.89%** | **96.91%** | 96.32% | 4.70% |
| mmBERT-base | 96.64% | 96.65% | **96.55%** | 5.58% |
| mmBERT-small | 95.94% | 95.95% | 81.13% | 6.24% |
| TF-IDF + Logistic Regression | 91.23% | 91.22% | 79.40% | 8.53% |
| XLM-RoBERTa base | 91.09% | 91.03% | 91.83% | 21.66% |
| DistilBERT German | 90.83% | 90.82% | 89.41% | 19.91% |
| Luna 6 zero-shot | 55.64% | 50.83% | — | **1.42%** |

The **Test Human false-positive rate** is the share of human-written test texts that were incorrectly classified as either AI-generated or AI-assisted.

The OOD evaluation shows a clear difference in generalization. mmBERT-base and GeLECTRA-large retain almost all of their held-out performance, while mmBERT-small drops from **95.94%** test accuracy to **81.13%** OOD accuracy.

### Learning curves

Each supervised model was trained on **1%, 5%, 12%, 17.5%, 25%, 50%, and 100%** of the training data. The following table shows validation accuracy at 100% of the training data.

| Model | Validation Accuracy |
|---|---:|
| **mmBERT-base** | **96.25%** |
| GeLECTRA-large | 95.84% |
| mmBERT-small | 94.84% |
| XLM-RoBERTa base | 91.60% |
| TF-IDF + Logistic Regression | 91.33% |
| DistilBERT German | 90.50% |

These values come from the learning-curve experiments and therefore represent **validation performance**, not the final held-out test results.

### Error rate per class

| Model | Human error | AI-generated error | AI-assisted error |
|---|---:|---:|---:|
| GeLECTRA-large | 4.70% | 1.54% | 3.07% |
| mmBERT-base | 5.58% | 0.99% | 3.51% |
| mmBERT-small | 6.24% | 0.99% | 4.94% |
| TF-IDF + Logistic Regression | 8.53% | 3.40% | 14.38% |
| XLM-RoBERTa base | 21.66% | 0.88% | 4.17% |
| DistilBERT German | 19.91% | 2.08% | 5.49% |
| Luna 6 zero-shot | 1.42% | 40.57% | 91.22% |


### Error rate by text length

The following table reports overall classification error across all three classes for different text-length bins.

`n` denotes the number of held-out test samples in each bin.

| Text length | n | Luna 6 | TF-IDF + Logistic Regression | DistilBERT German | XLM-RoBERTa base | mmBERT-small | mmBERT-base | GeLECTRA-large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ≤70 | 399 | 44.11% | 23.06% | 16.79% | 17.04% | 12.03% | 10.53% | **9.27%** |
| 71–150 | 293 | 39.59% | 8.19% | 9.90% | 8.87% | 4.44% | 3.41% | **2.73%** |
| 151–300 | 640 | 42.34% | 7.03% | 7.81% | 8.28% | 4.38% | 3.12% | **2.34%** |
| 301–500 | 342 | 48.83% | 5.26% | 6.14% | 5.56% | 1.17% | **0.58%** | 1.75% |
| 501–750 | 653 | 44.87% | 5.82% | 7.50% | 6.43% | 1.99% | **1.53%** | 2.14% |
| 750+ | 410 | 46.59% | 5.61% | 8.54% | 8.78% | **1.22%** | 1.95% | **1.22%** |

The clearest pattern is that **very short texts are substantially harder to classify**. All supervised models show their highest error rates on texts of 70 words or fewer.


## Key Findings

- **GeLECTRA-large achieved the strongest in-distribution performance**, with the highest accuracy and Macro-F1 on the held-out test set.

- **mmBERT-base showed the strongest generalization to unseen data**, maintaining nearly the same performance on the out-of-distribution evaluation set.

- **High test-set accuracy did not always translate into robust generalization.** In particular, mmBERT-small performed strongly on the original test set but showed a much larger performance drop on unseen-source data.

- **Short texts were consistently more difficult to classify** than medium- and long-length texts across the evaluated models.

- **AI-assisted texts were harder to distinguish than fully AI-generated texts**, indicating that partial human authorship makes detection substantially more challenging.

- **Increasing model complexity generally improved performance, but architecture choice mattered more than size alone.** The experiments showed substantial differences in robustness between Transformer models with otherwise similar in-distribution results.


## Limitations

- The models were trained primarily on **German academic texts**, mainly from the social sciences and education research, and on text generated by **four LLM families**. Performance may therefore differ for other domains, writing styles, languages, or newer AI models.

- **AI-assisted writing can take many forms.** In this project, it refers to a specific setting in which the complete human-written text is rewritten and polished by an AI model while preserving its content and structure.

- Inputs are truncated to **512 tokens**, meaning that for longer texts only the beginning of the document is used for classification.

- The detector produces **probabilistic, not proof**. Its output should not be used on its own to accuse, penalize, grade, or otherwise make high-stakes decisions about individuals.




## Getting started

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/HassanHijaze/german-ai-text-detector-from-scratch.git
cd german-ai-text-detector-from-scratch
pip install -r requirements.txt

```
Copy `.env.example` to `.env` and add the credentials required for the stages you want to run. API keys are needed for the AI-generation and external API stages. The dataset is private, so stages that load it from the Hugging Face Hub require access permission; the trained model repositories are public. Each stage contains its own README with specific setup and execution instructions.

## Authors

- [Hassan Hijazi](https://github.com/HassanHijaze)
- [Ali Issa](https://github.com/aliissa824)


## License

The source code in this repository is released under the [MIT License](LICENSE).

Datasets, source texts, pretrained models, and other third-party resources remain subject to their respective original licenses and terms of use.
