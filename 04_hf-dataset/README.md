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

German academic texts in three classes: written by humans, polished by AI, 
or written by AI.

The human texts are openly licensed papers, theses and book chapters 
published between 1990 and 2022, before AI writing tools were widely used. 
For each human text, an AI model wrote two partners: an **AI-assisted** 
version (the human text rewritten and polished) and an **AI-generated** 
text (a new text on the same topic).

## Labels

| label | name | meaning |
|---|---|---|
| 0 | `human` | Original human-written text |
| 1 | `ai_generated` | A new text on the same topic, written by an AI model |
| 2 | `ai_assisted` | The human text, rewritten and polished by an AI model |

## Size

| Split | Human | AI-generated | AI-assisted | Total |
|---|---:|---:|---:|---:|
| Train | 8,342 | 8,324 | 8,320 | 24,986 |
| Validation | 971 | 967 | 968 | 2,906 |
| Test | 914 | 912 | 911 | 2,737 |
| **Total** | **10,227** | **10,203** | **10,199** | **30,629** |

All texts made from the same human original are in the same split, and no 
source document appears in more than one split.

## How it was made

- **Human texts:** chunks of 35–1,050 words (mostly 50–1,000) that start and end on full 
sentences, from SSOAR (social sciences), peDOCS (education research) and 
Refubium (FU Berlin). Only open licences.
- **AI texts:** GPT-5.6 Luna, GPT-6 Luna, Gemini 3.8 Flash and DeepSeek V4 
Flash. Each model handled about a quarter of the human texts.
- **AI-generated strategy:** GPT and Gemini saw the human text as 
background material. DeepSeek only saw 3 topics extracted from it (see 
[02 · AI data](../02_ai-generated-data/)).


## Fields

| field | description |
|---|---|
| `id` | Unique integer: human rows 1–10,227, AI-generated rows 10,228–20,436, AI-assisted rows 20,437–30,635. An AI row's `id` is not its human original's `id`; use `seed_sample_id` |
| `file` | Path of the source file from the original generation run (not verified; folder names don’t always match the label) |
| `text` | The text |
| `label` | 0 = human, 1 = ai_generated, 2 = ai_assisted (an integer, not a class label) |
| `seed_sample_id` | For AI rows: `id` of the human original (empty for human rows) |
| `language`, `split` | Always `de`; `train`, `validation` or `test` |
| `source`, `source_name`, `source_type`, `document_id` | Repository and original document |
| `title`, `document_type`, `publication_date`, `date_basis` | The original document, and where its date comes from |
| `license_name`, `license_url`, `source_url` | Licence and link to the original |
| `target_length`, `original_document_length` | Planned length of the human chunk (50, 150, 300, 600 or 1,000 words) and length of the original document (AI rows copy both from their human original) |
| `actual_word_count`, `character_count` | Actual length of the text in words and characters |
| `retrieval_date`, `hash` | When the original was downloaded, and the SHA-256 hash of the text |
| `generator_provider`, `generator_model` | Which company and model made an AI row (empty for human rows) |

This describes `data/hf-dataset-3class-full`, the dataset used to train and evaluate every model in this project. `build_hf_dataset.ipynb` in this folder builds a different version (`data/hf-dataset`, with string IDs such as `human-46` and `assisted-46`, and `author`, `publication_year`, `word_count`, `generator_company` and `strategy` columns).

## Load it

```python
from datasets import load_from_disk

ds = load_from_disk("data/hf-dataset-3class-full")
LABELS = {0: "human", 1: "ai_generated", 2: "ai_assisted"}  # label is a plain integer
row = ds["train"][0]
print(row["label"], LABELS[row["label"]])  # 0 human
```

## Licence and attribution

Released under **CC BY-SA 4.0**, because part of the human texts are CC 
BY-SA. Every row keeps the title, licence and source URL of the 
original work it is based on.

## Limitations

- Mostly social sciences and education research (SSOAR and peDOCS make up 
about 91% of the human texts).
- Published, edited academic texts, not student essays.
- Each AI model wrote texts for a different set of human texts.

