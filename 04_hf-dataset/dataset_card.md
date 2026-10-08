---
language:
- de
license: cc-by-sa-4.0
task_categories:
- text-classification
tags:
- ai-generated-text-detection
- german
- academic-writing
size_categories:
- 10K<n<100K
pretty_name: German Academic AI Text Detection
---

# German Academic AI Text Detection

German academic texts in three classes: written by humans, polished by AI, or written by AI.

The human texts are openly licensed papers, theses and book chapters published between 1990 and 2022, before AI writing tools were widely used. For each human text, an AI model wrote two partners: an **AI-assisted** version (the human text rewritten and polished) and an **AI-generated** text (a new text on the same topic).

## Labels

| label | name | meaning |
|---|---|---|
| 0 | `human` | Original human-written text |
| 1 | `ai_assisted` | The human text, rewritten and polished by an AI model |
| 2 | `ai_generated` | A new text on the same topic, written by an AI model |

## Size

| split | human | ai_assisted | ai_generated | total |
|---|---|---|---|---|
{{COUNTS_TABLE}}

All texts made from the same human original are in the same split, and no source document appears in more than one split.

## How it was made

- **Human texts:** chunks of 50–1,000 words that start and end on full sentences, from SSOAR (social sciences), peDOCS (education research) and Refubium (FU Berlin). Only open licences.
- **AI texts:** GPT-5.6 Luna, GPT-6 Luna, Gemini 3.8 Flash and DeepSeek V4 Flash. Each model handled about a quarter of the human texts.
- **AI-generated strategy:** GPT and Gemini saw the human text as background material. DeepSeek only saw 3 topics extracted from it (see the `strategy` column).

The full pipeline and code: https://github.com/aliissa824/german-ai-text-detector

## Fields

| field | description |
|---|---|
| `id` | `human-<n>`, `assisted-<n>` or `generated-<n>`, where `<n>` is the human original's ID |
| `text` | The text |
| `label` | 0 = human, 1 = ai_assisted, 2 = ai_generated |
| `seed_sample_id` | For AI rows: ID of the human original (empty for human rows) |
| `source`, `source_name`, `document_id` | Repository and original document |
| `title`, `author`, `publication_year`, `document_type` | The original document |
| `license_name`, `license_url`, `source_url` | Licence and link to the original |
| `target_length`, `word_count` | Planned and actual length in words |
| `generator_model`, `generator_company`, `strategy` | Which model made an AI row, and how (`rewrite`, `source_based`, `topics`) |

## Load it

```python
from datasets import load_dataset

ds = load_dataset("{{HF_REPO_ID}}")
print(ds["train"].features["label"].names)  # ['human', 'ai_assisted', 'ai_generated']
```

## Licence and attribution

Released under **CC BY-SA 4.0**, because part of the human texts are CC BY-SA. Every row keeps the author, title, licence and source URL of the original work it is based on.

## Limitations

- Mostly social sciences and education research (SSOAR and peDOCS make up about 91% of the human texts).
- Published, edited academic texts, not student essays.
- Each AI model wrote texts for a different set of human texts.
