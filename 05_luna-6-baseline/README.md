[← 04 · Hugging Face dataset](../04_hf-dataset/)

# 05 · Luna 6 Zero-Shot Baseline

Can a general-purpose LLM distinguish **Human**, **AI-generated**, and **AI-assisted** German academic texts without task-specific training?

To test this, **GPT-6 Luna** classifies the complete held-out test set using only a definition of the three classes. No examples are provided and no fine-tuning is performed.

The same **2,737 test samples** are used for the supervised models later in the project, making Luna 6 a direct zero-shot baseline.

## Setup

| | |
|---|---|
| Model | GPT-6 Luna |
| API | OpenAI API |
| Task | Three-class classification |
| Classes | Human, AI-generated, AI-assisted |
| Test samples | 2,737 |
| Human | 914 |
| AI-generated | 912 |
| AI-assisted | 911 |
| Training | None |
| Prompting | Zero-shot |

For each text, Luna 6 returns a predicted label and a probability for each of the three classes.

## Results

All **2,737 test samples** were classified successfully.

| **Metric** | **Result** |
|---|---:|
| Accuracy | **55.64%** |
| Macro-F1 | **50.83%** |
| Human error | **1.42%** |
| AI-generated error | **40.57%** |
| AI-assisted error | **91.22%** |



Luna 6 recognizes **Human text very reliably**, correctly classifying 901 of 914 samples.

Performance is weaker for **AI-generated text**. The model correctly identifies 542 of 912 samples, while 304 are classified as AI-assisted.

**AI-assisted text is the main failure case.** Only 80 of 911 samples are classified correctly, while 803 are predicted as Human.

Overall, the zero-shot model strongly favors the Human class and struggles to identify AI-assisted writing.


## Results by Text Length

Performance was also evaluated across six text-length ranges.

| **Words** | **Samples** | **Accuracy** | **Human error** | **AI-generated error** | **AI-assisted error** |
|---|---:|---:|---:|---:|---:|
| ≤70 | 399 | 55.89% | 6.72% | 37.31% | 89.31% |
| 71–150 | 293 | 60.41% | 1.06% | 41.80% | 83.12% |
| 151–300 | 640 | 57.66% | 0.48% | 40.00% | 91.49% |
| 301–500 | 342 | 51.17% | 1.89% | 35.96% | 90.48% |
| 501–750 | 653 | 55.13% | 0.00% | 43.98% | 91.27% |
| 750+ | 410 | 53.41% | 0.00% | 41.98% | 97.84% |

Overall accuracy remains between approximately **51% and 60%** across all length ranges.

Human error decreases for longer texts, while AI-assisted error remains very high across every range. Text length therefore does not solve the main classification problem.




## How to Run

Add your OpenAI API key to the `.env` file in the repository root:

```env
OPENAI_API_KEY=your_api_key_here
```

Make sure the dataset created in step 04 is available at:

```text
data/hf-dataset-3class-full
```

Then open:

```text
GPT_baseline.ipynb
```

and run the notebook cells in order.


---

[← 04 · Hugging Face dataset](../04_hf-dataset/) · [06 · TF-IDF baseline →](../06_logreg-baseline/)