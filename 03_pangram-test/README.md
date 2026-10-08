[← 02 · AI data](../02_ai-generated-data/)

# 03 · Pangram check

Before training anything, we wanted to know: **are our labels actually right?**

To check, we took **100 random samples** (50 human, 50 AI-generated) and ran them through [Pangram](https://www.pangram.com). **Pangram classified all 100 correctly.**

## Output format

`pangram_results.csv` has one row per checked sample. Ten real examples:

| sample_id | true_label | model | word_count | pangram_prediction |
|---|---|---|---|---|
| 4378 | Human | – | 35 | Human |
| 9682 | Human | – | 141 | Human |
| 6062 | Human | – | 304 | Human |
| 344 | Human | – | 603 | Human |
| 7157 | Human | – | 1008 | Human |
| 371 | AI | GPT-5.6 Luna | 504 | AI |
| 2510 | AI | Gemini 3.8 Flash | 488 | AI |
| 6550 | AI | DeepSeek V4 Flash | 288 | AI |
| 8165 | AI | GPT-6 Luna | 168 | AI |
| 7641 | AI | GPT-6 Luna | 1143 | AI |

For AI rows, `sample_id` is the ID of the human text the AI text was made from.

## Results

| | Human | AI |
|---|---|---|
| **Human** (50) | 50 | 0 |
| **AI-generated** (50) | 0 | 50 |

**Every model was caught:**

| Model | AI samples checked | Detected as AI |
|---|---|---|
| GPT-5.6 Luna | 14 | 14 |
| GPT-6 Luna | 14 | 14 |
| DeepSeek V4 Flash | 13 | 13 |
| Gemini 3.8 Flash | 9 | 9 |

**Short texts too:** The samples range from 35 to 1,143 words, and 19 of them are 100 words or shorter. Those were all classified correctly as well.

## What this tells us

- **The human data is clean.**
- **The AI-generated data really reads as AI.** 
- **It's a sample, not a full test.** 100 out of 100 doesn't mean Pangram is perfect, but it's a strong sign. With 100 correct out of 100, the true error rate on this kind of data is very likely below 3%.

**Not tested here: AI-assisted texts.** This check only covers human vs. AI-generated. AI-assisted texts are the harder case, since they are human writing polished by AI, and they are where our own baseline makes most of its mistakes.

## Pangram was also used during generation

Before each full generation run in step 02, we tested a small sample from each model on Pangram. That's how we found that DeepSeek's first prompt produced texts that were partly paraphrased from the human original, and switched it to a topics-only prompt. See [02 · AI data](../02_ai-generated-data/) for the details.

---

[← 02 · AI data](../02_ai-generated-data/) -- [04 · Hugging Face dataset →](../04_hf-dataset/)
