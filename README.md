# German AI Text Detector

A three-class research project for distinguishing **human-written**, **AI-generated**, and **AI-assisted** German academic text.

The repository covers the full pipeline: data collection, AI-text generation, dataset construction, model training, error analysis, learning curves, and cross-model comparison.

Unlike a binary Human-vs-AI detector, this project treats **AI-assisted writing** (human text rewritten or polished by a language model) as a separate class.

> **Status:** the dataset pipeline, six supervised models with held-out test results, a zero-shot LLM baseline, length-bias analysis, and learning curves are complete.


## Repository guide

| Stage | Folder | Purpose |
| ---: | --- | --- |
| 01 | [`01_human-data`](./01_human-data/) | Collect and clean German academic human text |
| 02 | [`02_ai-generated-data`](./02_ai-generated-data/) | Generate AI and AI-assisted counterparts |
| 03 | [`03_pangram-test`](./03_pangram-test/) | External Human-vs-AI sanity check |
| 04 | [`04_hf-dataset`](./04_hf-dataset/) | Build the final Hugging Face dataset |
| 05 | [`05_logreg-baseline`](./05_logreg-baseline/) | TF-IDF + Logistic Regression baseline |
| 06 | [`06_mmBERT-small`](./06_mmBERT-small/) | mmBERT-small experiments |
| 07 | [`07_distilbert-base-german-cased`](./07_distilbert-base-german-cased/) | DistilBERT German experiments |
| 08 | [`08_mmbert-base`](./08_mmbert-base/) | mmBERT-base experiments |
| 09 | [`09_xlm-roberta-base`](./09_xlm-roberta-base/) | XLM-RoBERTa base experiments |
| 10 | [`10_Gelectra-large`](./10_Gelectra-large/) | GeLECTRA-large experiments |
| 11 | [`11_gpt`](./11_gpt/) | Luna 6 zero-shot baseline |
| 12 | [`12_Learning-curve`](./12_Learning-curve/) | Learning-curve experiments |
| 13 | [`13_model-comparison`](./13_model-comparison/) | Cross-model comparison |


Each stage has its own README with methods, code, and outputs.



## At a glance

- **30,629 texts** in the final three-class dataset
- **10,227 human texts** from **4,121 source documents** (SSOAR, peDOCS, Refubium)
- AI and AI-assisted texts from **four generator families**
- Document-level train / validation / test splits
- **Six supervised classifiers** and **one zero-shot LLM baseline**
- Evaluation with confusion matrices, class-specific results, length analysis, and learning curves

## Dataset

### Human texts

| Source   | Type                                          |
| -------- | --------------------------------------------- |
| SSOAR    | Open-access repository for social sciences    |
| peDOCS   | Open-access repository for education research |
| Refubium | Freie Universität Berlin repository           |

Only openly licensed material was kept (CC BY, CC BY-SA, CC0, Public Domain). The texts come from German academic documents published before the widespread use of generative AI. Source and license information is stored per sample.

### Text construction

Documents were cut into **sentence-complete chunks** with target lengths of about 50, 150, 300, 600, and 1,000 words. At most **three chunks per document** were used, and all chunks of one document stay in the same split.

### AI-generated and AI-assisted texts

For most human samples, an AI-generated and an AI-assisted counterpart were created with the prompts in this repository.

- **AI-assisted:** the model receives the human text and rewrites or polishes it while preserving meaning, argument, order of ideas, citations, and numbers.
- **AI-generated:** the model writes a new academic text on the same topic with its own wording and structure.
- **DeepSeek exception:** the first DeepSeek setup paraphrased the human source too closely. For DeepSeek, the final AI-generated strategy first extracts three short topics from the human text and then generates a new text from those topics only.

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

Related texts (the same source document and everything derived from it) always stay in the same split, which reduces source leakage.

## Models

| Model                        | Type                   |
| ---------------------------- | ---------------------- |
| TF-IDF + Logistic Regression | Classical baseline     |
| DistilBERT German            | Transformer            |
| mmBERT-small                 | Transformer            |
| mmBERT-base                  | Transformer            |
| XLM-RoBERTa base             | Transformer            |
| GeLECTRA-large               | Transformer            |
| Luna 6 (via Azure OpenAI)    | Zero-shot LLM baseline |


## Results

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

These are **validation** results from the learning-curve experiments, not final held-out test results.

### Final results

The six supervised models were evaluated on the held-out test set of 2,737 texts (914 Human, 912 AI-generated, 911 AI-assisted). Luna 6 was evaluated separately on a balanced sample of 9,000 texts. "Human flagged as AI or AI-assisted" is the share of human texts predicted as anything other than Human.

| Model                        |   Accuracy |   Macro-F1 | Human flagged as AI or AI-assisted |
| ---------------------------- | ---------: | ---------: | ---------------------------------: |
| **GeLECTRA-large**           | **96.89%** | **96.91%** |                          **4.70%** |
| mmBERT-base                  |     96.64% |     96.65% |                              5.58% |
| mmBERT-small                 |     95.98% |     95.98% |                              5.47% |
| TF-IDF + Logistic Regression |     91.23% |     91.22% |                              8.53% |
| XLM-RoBERTa base             |     91.09% |     91.03% |                             21.66% |
| DistilBERT German            |     90.83% |     90.82% |                             19.91% |
| Luna 6 zero-shot*            |     57.03% |     54.96% |                              4.78% |

\* Luna 6 was not fine-tuned and was evaluated on a separate balanced sample of 9,000 texts, so its result is shown as a reference rather than a directly comparable test score.

### Error rate per class

| Model                        | Human error | AI-generated error | AI-assisted error |
| ---------------------------- | ----------: | -----------------: | ----------------: |
| GeLECTRA-large               |       4.70% |              1.54% |             3.07% |
| mmBERT-base                  |       5.58% |              0.99% |             3.51% |
| mmBERT-small                 |       5.47% |              0.66% |             5.93% |
| TF-IDF + Logistic Regression |       8.53% |              3.40% |            14.38% |
| XLM-RoBERTa base             |      21.66% |              0.88% |             4.17% |
| DistilBERT German            |      19.91% |              2.08% |             5.49% |
| Luna 6 zero-shot*            |       4.78% |             47.53% |            76.53% |



### External sanity check (Pangram)

A random sample of 50 human and 50 AI-generated texts, from about 35 to 1,143 words and covering all four generators, was checked with Pangram. All 100 were classified correctly. The sample is small and AI-assisted texts were not included, so this is a sanity check on the data and not a benchmark.

## Key findings

1. **The top three models are close.** GeLECTRA-large (96.89%), mmBERT-base (96.64%), and mmBERT-small (95.98%) lead on the test set. The gap between the first two is 0.25 percentage points, about 7 of 2,737 texts, so their order should not be over-interpreted. On the validation curves mmBERT-base was ahead (96.25% vs 95.84%).
2. **A small model is competitive.** mmBERT-small (95.98%) beats XLM-RoBERTa base, DistilBERT German, and the baseline by about 5 points, and is within about 1 point of GeLECTRA-large. Model size alone does not decide performance.
3. **Accuracy hides a large difference in human error.** XLM-RoBERTa base and DistilBERT German reach about 91% accuracy but flag roughly 1 in 5 human texts as AI or AI-assisted (21.66% and 19.91%). The TF-IDF baseline has the same accuracy (91.23%) and flags 8.53%. The three best models flag between 4.7% and 5.6%.
4. **Fine-tuning is needed.** The zero-shot Luna 6 baseline reaches 57.03% accuracy. It rarely flags human text (4.78%, similar to the best fine-tuned models) but misses 47.53% of AI-generated and 76.53% of AI-assisted texts, so it behaves conservatively. This result depends on the prompt used and was measured on a different 9,000-text sample, so it is a rough comparison only.
5. **Fully AI-generated text is the easiest class** for every supervised model (0.66% to 3.40% error). The hardest class differs by model: for the baseline and mmBERT-small it is AI-assisted text (14.38% and 5.93%), while for the other four it is human text.
6. **The baseline struggles most with AI-assisted text** (14.38% error), which keeps the human content and structure by construction. Fine-tuned transformers reduce this to 3.07% to 5.93%.
7. **Generation method matters.** The Pangram check exposed that the original DeepSeek setup produced near-paraphrases of the human source, which led to the topics-only strategy.
8. **Validation and test results agree.** The model order is similar and accuracies differ by at most about 1.1 points.
9. **Text length.** Results were analyzed in the bins <=70, 71-150, 151-300, 301-500, 501-750, and 750+ words.

## Evaluation

The repository reports more than accuracy: three-class confusion matrices, class-specific performance, prediction confidence, length-bias analysis, training vs. validation accuracy (generalization gap), learning curves, and the zero-shot LLM comparison.

## Limitations

- **Domain.** All texts are German academic writing. Results should not be assumed to transfer to other genres, registers, or languages.
- **Generation prompts.** The AI texts were produced with prompts asking for polished academic German. Style and polish may therefore act as a signal in addition to authorship. A test on plainer or differently styled text from other generators is planned.
- **Generators.** Four model families are covered, not every current or future model.
- **AI-assisted class.** Real AI-assisted writing varies widely. This class represents one specific form of AI rewriting and polishing.
- **Short texts.** Texts of 50 words or fewer are harder to classify.
- **Sanity check.** The Pangram test is small and excludes AI-assisted text.



## Reproduce

Install the required packages:

```bash
pip install -r requirements.txt