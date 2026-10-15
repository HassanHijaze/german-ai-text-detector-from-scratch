# German AI Text Detector

A three-class machine learning project for distinguishing **human-written**, **AI-generated**, and **AI-assisted** German text.

The project covers the complete pipeline from data collection and AI-text generation to dataset construction, model training, learning curves, error analysis, and out-of-distribution evaluation.

Unlike conventional binary Human-vs-AI detectors, this project treats **AI-assisted writing** — human text rewritten or polished by a language model — as a separate class.

> **Top held-out result:** GeLECTRA-large reaches **96.89% accuracy** and **96.91% Macro-F1**, closely followed by mmBERT-base at **96.64% accuracy**.  
> **Top OOD results:** mmBERT-base reaches **96.55% accuracy**, closely followed by GeLECTRA-large at **96.32%**, on 869 texts from previously unseen source domains.

## At a glance

- **30,629 texts** across three classes
- **10,227 human texts** from **4,121 academic source documents**
- AI-generated and AI-assisted texts from **four LLM families**
- Document-level train / validation / test splitting to reduce source leakage
- **6 supervised classifiers** and **1 zero-shot LLM baseline**
- Best held-out test accuracy: **96.89% — GeLECTRA-large**
- Best OOD accuracy: **96.55% — mmBERT-base**
- Evaluation includes **Macro-F1, confusion matrices, class-specific errors, text-length analysis, learning curves, calibration, and OOD generalization**




## Repository guide

| Stage | Folder | Purpose |
|---:|---|---|
| 01 | [`01_human-data`](./01_human-data/) | Collect and clean German academic human text |
| 02 | [`02_ai-generated-data`](./02_ai-generated-data/) | Generate AI and AI-assisted counterparts |
| 03 | [`03_pangram-test`](./03_pangram-test/) | External Human-vs-AI sanity check |
| 04 | [`04_hf-dataset`](./04_hf-dataset/) | Build the final Hugging Face dataset |
| 05 | [`05_logreg-baseline`](./05_logreg-baseline/) | TF-IDF + Logistic Regression baseline |
| 06 | [`06_mmBERT-small`](./06_mmBERT-small/) | mmBERT-small experiments |
| 07 | [`07_distilbert-german`](./07_distilbert-german/) | DistilBERT German experiments |
| 08 | [`08_mmbert-base`](./08_mmbert-base/) | mmBERT-base experiments |
| 09 | [`09_xlm-roberta-base`](./09_xlm-roberta-base/) | XLM-RoBERTa-base experiments |
| 10 | [`10_Gelectra-large`](./10_Gelectra-large/) | GeLECTRA-large experiments |
| 11 | [`11_luna-6-baseline`](./11_luna-6-baseline/) | Luna-6 zero-shot baseline |
| 12 | [`12_learning-curves`](./12_learning-curves/) | Learning-curve experiments |
| 13 | [`13_model-comparison`](./13_model-comparison/) | Cross-model comparison |
| 14 | [`14_ood-model-comparison`](./14_ood-model-comparison/) | Out-of-distribution model comparison |



## Dataset

### Human texts

| Source   | Type                                          |
| -------- | --------------------------------------------- |
| SSOAR    | Open-access repository for social sciences    |
| peDOCS   | Open-access repository for education research |
| Refubium | Freie Universität Berlin repository           |

Only openly licensed material was kept (CC BY, CC BY-SA, CC0, Public Domain). The texts come from German academic documents published before the widespread use of generative AI. Source and license information is stored per sample.


### AI-generated and AI-assisted texts

For most human samples, an AI-generated and an AI-assisted counterpart were created with the prompts in this repository.

- **AI-assisted:** the model receives the human text and rewrites or polishes it while preserving meaning, argument, order of ideas, citations, and numbers.
- **AI-generated:** the model writes a new academic text on the same topic with its own wording and structure.





| Generator         | AI-generated | AI-assisted |
| ----------------- | -----------: | ----------: |
| GPT-5.6 Luna      |        2,491 |       2,489 |
| Gemini 3.8 Flash  |        2,494 |       2,493 |
| DeepSeek V4 Flash |        2,497 |       2,500 |
| GPT-6 Luna        |        2,721 |       2,718 |

### Class and split sizes

| Class        |    Samples |
| ------------ | ---------: |
| Human        |     10,227 |
| AI-generated |     10,203 |
| AI-assisted  |     10,199 |
| **Total**    | **30,629** |

| Split      | Human |    AI | AI-assisted |  Total |
| ---------- | ----: | ----: | ----------: | -----: |
| Train      | 8,342 | 8,324 |       8,320 | 24,986 |
| Validation |   971 |   967 |         968 |  2,906 |
| Test       |   914 |   912 |         911 |  2,737 |


### Out-of-distribution evaluation

To evaluate generalization beyond the academic training distribution, all six supervised models were tested on a separate dataset of **869 texts**:

- **290 Human**
- **289 AI-generated**
- **290 AI-assisted**

The human texts come from previously unseen domains:

- German Wikipedia — encyclopedic text
- German Wikinews — news
- German Wikivoyage — travel
- German Wikibooks — instructional text

The same four LLM families used in the main dataset were used to create the corresponding AI-generated and AI-assisted samples.



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
| Luna 6 (Azure OpenAI) | Zero-shot LLM baseline | Not publicly disclosed |



## Results

### Main model comparison

The supervised models were evaluated on the held-out academic test set and on a separate out-of-distribution (OOD) dataset containing 869 texts from previously unseen source domains.

| Model | Test Accuracy | OOD Accuracy | OOD Human vs AI Accuracy |
|---|---:|---:|---:|
| **mmBERT-base** | 96.64% | **96.55%** | **98.27%** |
| **GeLECTRA-large** | **96.89%** | 96.32% | 96.37% |
| XLM-RoBERTa base | 91.09% | 91.83% | 91.36% |
| DistilBERT German | 90.83% | 89.41% | 90.33% |
| mmBERT-small | 95.94% | 81.13% | 74.44% |
| TF-IDF + Logistic Regression | 91.23% | 79.40% | 96.89% |


### Learning curves

Each supervised model was trained on 1%, 5%, 12%, 17.5%, 25%, 50%, and 100% of the training data. Validation accuracy at 100%:

| Model                        | Validation accuracy |
| ---------------------------- | ------------------: |
| **mmBERT-base**              |          **96.25%** |
| GeLECTRA-large               |              95.84% |
| mmBERT-small                 |              94.84% |
| XLM-RoBERTa base             |              91.60% |
| TF-IDF + Logistic Regression |              91.33% |
| DistilBERT German            |              90.50% |


### Final results

The six supervised models were evaluated on the held-out test set of 2,737 texts (914 Human, 912 AI-generated, 911 AI-assisted). Luna-6 was evaluated separately on a balanced sample of approximately 9,000 texts. "Human flagged as AI or AI-assisted" is the share of human texts predicted as anything other than Human.

| Model | Accuracy | Macro-F1 | Human flagged as AI or AI-assisted |
|---|---:|---:|---:|
| **GeLECTRA-large** | **96.89%** | **96.91%** | **4.70%** |
| mmBERT-base | 96.64% | 96.65% | 5.58% |
| mmBERT-small | 95.94% | 95.95% | 6.24% |
| TF-IDF + Logistic Regression | 91.23% | 91.22% | 8.53% |
| XLM-RoBERTa base | 91.09% | 91.03% | 21.66% |
| DistilBERT German | 90.83% | 90.82% | 19.91% |
| Luna-6 zero-shot* | 57.03% | 54.96% | 4.78% |



### Error rate per class

| Model | Human error | AI-generated error | AI-assisted error |
|---|---:|---:|---:|
| GeLECTRA-large | 4.70% | 1.54% | 3.07% |
| mmBERT-base | 5.58% | 0.99% | 3.51% |
| mmBERT-small | 6.24% | 0.99% | 4.94% |
| TF-IDF + Logistic Regression | 8.53% | 3.40% | 14.38% |
| XLM-RoBERTa base | 21.66% | 0.88% | 4.17% |
| DistilBERT German | 19.91% | 2.08% | 5.49% |
| Luna-6 zero-shot* | 4.78% | 47.53% | 76.53% |


---

## Error rate by text length

The following table reports overall classification error across all three classes for different text-length bins.

`n` denotes the number of held-out test samples in each bin.

| Text length | n | TF-IDF + Logistic Regression | DistilBERT German | XLM-RoBERTa base | mmBERT-small | mmBERT-base | GeLECTRA-large |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤70 | 399 | 23.06% | 16.79% | 17.04% | 12.03% | 10.53% | **9.27%** |
| 71–150 | 293 | 8.19% | 9.90% | 8.87% | 4.44% | 3.41% | **2.73%** |
| 151–300 | 640 | 7.03% | 7.81% | 8.28% | 4.38% | 3.12% | **2.34%** |
| 301–500 | 342 | 5.26% | 6.14% | 5.56% | 1.17% | **0.58%** | 1.75% |
| 501–750 | 653 | 5.82% | 7.50% | 6.43% | 1.99% | **1.53%** | 2.14% |
| 750+ | 410 | 5.61% | 8.54% | 8.78% | **1.22%** | 1.95% | **1.22%** |

The clearest pattern is that **very short texts are substantially harder to classify**.



## Key findings


- **A zero-shot LLM is not enough.** Luna 6 reaches **57.03% accuracy** and **54.96% Macro-F1** without task-specific training, far below the supervised models, which achieve between **90.83% and 96.89% accuracy**.

- **A simple baseline goes surprisingly far.** TF-IDF + Logistic Regression reaches **91.23% test accuracy** without Transformer fine-tuning, showing that lexical patterns alone provide a strong baseline for this task.

- **Doing well on the test set isn't enough.** On the held-out academic test set, mmBERT-small (**95.94%**), mmBERT-base (**96.64%**) and GeLECTRA-large (**96.89%**) perform similarly. On texts from previously unseen source domains, mmBERT-base (**96.55%**) and GeLECTRA-large (**96.32%**) remain strong, while mmBERT-small drops to **81.13%**. OOD evaluation is what clearly separates their generalization performance.

- **AI-assisted text is harder than fully AI-generated text.** Across the supervised models, AI-generated text has very low error rates of **0.88%–3.40%**, while AI-assisted text produces higher error rates of **3.07%–14.38%**.

- **Short texts are substantially harder.** For texts of **70 words or fewer**, error rates range from **9.27% to 23.06%**, considerably higher than for longer texts.
The repository reports more than accuracy: three-class confusion matrices, class-specific performance, prediction confidence, length-bias analysis, training vs. validation accuracy (generalization gap), learning curves, and the zero-shot LLM comparison.

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


## Authors

- [Hassan Hijazi](https://github.com/HassanHijaze)
- [Ali Issa](https://github.com/aliissa824)


## License

The source code in this repository is released under the [MIT License](LICENSE).

Datasets, source texts, pretrained models, and other third-party resources remain subject to their respective original licenses and terms of use.
