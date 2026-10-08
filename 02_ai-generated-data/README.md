[← 01 · Human data](../01_human-data/)

# 02 · AI data

For every human sample from step 01, one AI model writes two texts:

| Class | What the model gets | What it's asked to do |
|---|---|---|
| **AI-assisted** | The human text | Rewrite and polish it: fix errors, improve wording and flow, but keep the meaning, argument, order of ideas, citations and numbers |
| **AI-generated** | The human text, or only 3 topics from it (see below) | Write a completely new academic text on the same topic, in its own words and structure |

Both texts are written by the same model for the same human sample, so almost every human text has an AI-assisted and an AI-generated partner. Both partners always land in the same split as their human original.

## Output format

Both AI texts are saved under the ID of the human text they came from: `data/ai-assisted/<seed_sample_id>.txt` and `data/ai-generated/<seed_sample_id>.txt`. Ten real examples:

| seed_sample_id | model | strategy | human words | AI-assisted words | AI-generated words |
|---|---|---|---|---|---|
| 697 | GPT-5.6 Luna | source-based | 41 | 43 | 38 |
| 1775 | GPT-5.6 Luna | source-based | 987 | 983 | 999 |
| 2375 | GPT-5.6 Luna | source-based | 598 | 643 | 529 |
| 2884 | Gemini 3.8 Flash | source-based | 152 | 156 | 145 |
| 4172 | Gemini 3.8 Flash | source-based | 295 | 305 | 284 |
| 7026 | DeepSeek V4 Flash | topics | 975 | 966 | 727 |
| 7164 | DeepSeek V4 Flash | topics | 285 | 296 | 276 |
| 7851 | GPT-6 Luna | source-based | 146 | 151 | 146 |
| 9472 | GPT-6 Luna | source-based | 53 | 56 | 49 |
| 9857 | GPT-6 Luna | source-based | 601 | 604 | 567 |

New runs also record every file in `data/generation_log.jsonl`: model, strategy, word counts, timestamp, and for DeepSeek the 3 extracted topics.

## Models

| Model | Company | Access | AI-generated | AI-assisted |
|---|---|---|---|---|
| GPT-5.6 Luna | OpenAI | Azure OpenAI | 2,491 | 2,489 |
| Gemini 3.8 Flash | Google | Gemini API | 2,494 | 2,493 |
| DeepSeek V4 Flash | DeepSeek | DeepSeek API | 2,497 | 2,500 |
| GPT-6 Luna | OpenAI | Azure OpenAI | 2,721 | 2,718 |
| **Total** | | | **10,203** | **10,200** |

Each model worked on its own share of the human samples. A few requests failed for good, so 24 human samples have no AI-generated partner and 27 no AI-assisted one.

## Things we ran into

### Testing every model on Pangram first

Before each full run, we generated a small sample with the model and checked it with [Pangram](https://www.pangram.com), a commercial AI detector. The goal was to make sure the "AI-generated" texts really read as AI-written.

The first version of the AI-generated prompt gave the model the full human text and asked for a new text on the same topic. For GPT and Gemini, Pangram rated the results as AI-generated. For DeepSeek, it didn't: Pangram flagged parts of the texts as human. DeepSeek was mostly paraphrasing the text it saw, keeping much of its structure and wording.

So DeepSeek got a two-step prompt instead:

1. Read the human text and return only **3 short topics** (at most 10 words each).
2. Write a new academic text from **those topics only**, without ever seeing the original.

With this, Pangram rated DeepSeek's texts as fully AI-generated. As a result, the AI-generated class was made in two ways:

| Strategy | Models | Model sees the human text? |
|---|---|---|
| Source-based | GPT-5.6 Luna, GPT-6 Luna, Gemini 3.8 Flash | Yes, as background material |
| Topics only | DeepSeek V4 Flash | No, only 3 topics extracted from it |

The AI-assisted prompt is the same for all four models.

### Keeping the length close to the human text

The AI-generated prompt asks for the same length as the human text, ±20%. Most models land a bit shorter. The median AI-generated text is 91% of its human original's length for GPT-5.6 Luna and Gemini 3.8 Flash, 95% for GPT-6 Luna, and 99% for DeepSeek.

### Failed and broken answers

Long runs hit rate limits, empty answers and, once, a Gemini account that ran out of credits halfway through. The code skips files that already exist, so a run can be restarted and only fills the gaps.

20 requests (14 AI-assisted, 6 AI-generated) were refused: instead of a text, the model answered with a one-line refusal like *"I'm sorry, but I cannot assist with that request."* These are not part of the dataset. The code now rejects answers shorter than half the human text and retries them.

## How to run

Put your API keys in the `.env` file in the repo root (see `.env.example`). Then open `generate_ai_data.ipynb`, run one model at a time, and use `find_missing()` to see what's left.

All prompts are in `ai_generation/prompts.py`, exactly as used for the dataset. Models, ID ranges and prompt strategies are in `ai_generation/config.py`.

---

[← 01 · Human data](../01_human-data/) -- [03 · Pangram Test →](../03_pangram-test/)