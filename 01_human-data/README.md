# 01 · Human data

The human side of the detector: **10,227 samples of German academic writing** from 4,121 documents, all published between 1990 and 2022, before AI writing tools were widely used.

Every text comes from an open repository, is openly licensed, and was cleaned and cut into chunks of 50 to 1,000 words that always start and end on a full sentence.

## Output format

`data/raw/human/meta.json` lists every sample, and each text is saved as `data/raw/human/human/<sample_id>.txt`. Ten real examples:

| sample_id | source | year | document_type | target_length | actual_word_count | license_name |
| --- | --- | --- | --- | --- | --- | --- |
| 738 | ssoar | 2000 | journal article | 1000 | 1016 | CC BY-SA |
| 750 | ssoar | 2017 | Rezension | 150 | 168 | CC BY-SA |
| 2533 | ssoar | 1996 | journal article | 50 | 40 | CC BY |
| 4507 | refubium | 2020 | doc-type:article | 300 | 298 | CC BY |
| 4702 | refubium | 2021 | doc-type:article | 300 | 309 | CC BY |
| 5059 | refubium | 2019 | doc-type:book | 600 | 608 | CC BY |
| 5091 | pedocs | 2019 | Article | 150 | 149 | CC BY |
| 5322 | pedocs | 2021 | Article | 1000 | 978 | CC BY |
| 7870 | pedocs | 2021 | BookPart | 50 | 44 | CC BY-SA |
| 8296 | ssoar | 2006 | Monographie | 600 | 426 | CC BY-SA |

Each sample also stores `file`, `split`, `title`, `label` (`human`, turned into `0` in step 03), `source_name`, `source_type`, `document_id`, `author`, the full `publication_date` and its `date_basis`, `character_count`, `context_before` (~120 words before the chunk), `spelling_modernized`, `license_url`, `source_url`, `hash` (SHA-256 of the text, used to find duplicates) and `retrieval_date`.

## At a glance

| | |
|---|---|
| Samples | 10,227 |
| Documents | 4,121 |
| Authors | about 5,200 |
| Words | about 4.2 million (median 303 per sample) |
| Publication years | 1990–2022 |
| Sources | SSOAR, peDOCS, Refubium (FU Berlin) |
| Licences | open only (CC BY, CC BY-SA, CC0, Public Domain) |
| Collected | 24–25 September 2026 |

## Sources

| Source | Field | Samples | Share | Documents | Years |
|---|---|---|---|---|---|
| SSOAR | Social sciences | 5,972 | 58.4% | 2,390 | 1990–2022 |
| peDOCS | Education research | 3,353 | 32.8% | 1,342 | 2000–2022 |
| Refubium (FU Berlin) | All disciplines, incl. dissertations | 902 | 8.8% | 389 | 1996–2022 |

We aimed for 6,000 samples per source. German + academic + openly licensed turned out to be a small pool, so we stopped each source once it had little usable material left. That's why the sources are uneven.

**By decade**

| 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|
| 149 (1.5%) | 1,041 (10.2%) | 4,684 (45.8%) | 4,353 (42.6%) |

**By document type:** mostly journal articles (59%), followed by book chapters (15%), books (10%) and reviews (8%).

## From PDF to sample

1. **Find documents.** We harvest each repository's metadata (OAI-PMH) and keep only records that are German, openly licensed, published 1990–2022, and running text. Datasets, images, audio and video, software, lectures and patents are skipped.
2. **Download the PDF.** Encrypted files, files over 100 MB and files over 400 pages are skipped.
3. **Extract the body text.** We use font sizes to keep only the main text, which drops footnotes, headings and captions. We also remove cover pages, tables of contents, reference lists and English paragraphs.
4. **Repair PDF damage.** Words split by line-break hyphens, soft hyphens, ligatures (ﬁ → fi), footnote numbers glued to words, and broken spacing around punctuation. Documents whose PDF lost its spaces or character encoding are rejected entirely.
5. **Modernize old spelling** (see below).
6. **Cut into chunks** that start and end on a full sentence (see below).
7. **Quality check every chunk.** A chunk is dropped if it starts mid-sentence, looks like a reference list, a bullet list or table fragments, contains URLs, e-mails or DOIs, or still has broken characters or hyphenation.
8. **Assign a split** per document (see below).

## Things we ran into

### Cut-off sentences would have been a shortcut

Our first version cut each text at exactly the target length, for example after 100 words. Almost always, that lands in the middle of a sentence.

That's a problem, because AI models always finish their sentences. If the human texts end mid-sentence and the AI texts don't, the detector doesn't need to learn anything about writing style. It just checks whether the last sentence is finished. It would score great on our data and fail on real texts.

So chunks are now built sentence by sentence. When a chunk reaches its target length, we look at the sentence end just before the target and the one just after, and pick whichever is closer (on a tie, the longer one). The chunk is only kept if it lands within **±30%** of the target. Otherwise that chunk is skipped, which usually happens at the end of a document or with one huge "sentence" such as a flattened table.

This means the word count rarely matches the target exactly, and that's on purpose: only 3.5% of samples match their target exactly, but 68.5% are within ±5% and all but two within ±30%. Overall, 53.2% of chunks are a bit shorter than their target and 43.3% a bit longer.

### Not every period ends a sentence

To end chunks on full sentences, we first need to know where a sentence ends. A sentence ends with `.`, `!` or `?`, but in academic German a period very often means something else:

| Case | Example | How we handle it |
|---|---|---|
| Abbreviations | `Dr.`, `z. B.`, `vgl.`, `bzw.`, `Hrsg.` | A list of 93 common German (academic) abbreviations never counts as a sentence end |
| Initials and page numbers | `A. Müller`, `S. 45` | A single letter followed by a period is not an end |
| Letter chains | `z.B.`, `u.a.`, `d.h.` | Not an end |
| Decimal numbers | `3.14` | Not an end |
| Ordinal numbers | `im 19. Jahrhundert`, `am 3. Mai`, `die 2. Auflage` | A number with a period, followed by one of 43 words like *Jahrhundert*, *Mai*, *Auflage*, *Kapitel*, is not an end. But `..., S. 45. Danach` still is. |
| Lowercase next word | `... ca. zehn Jahre` | If the next word starts lowercase, the sentence goes on |
| Repeated marks | `...`, `?!` | Only the last mark counts |
| `?` or `!` inside a sentence | `(... widerlegend?) zum ...` | Not an end if the text continues in lowercase |

**Quotes and brackets.** When a sentence ends inside a quote, like `„… legte ich das Abitur ab.“`, the closing quote mark (or bracket) after the period is kept with that sentence. Otherwise the next chunk would start with a stray `“`, and the previous one would end on an unclosed quote. Any unfinished text at the very end of a document is dropped.

### Old spelling would have been another shortcut

Texts from before the 1996 spelling reform write `daß`, `Prozeß`, `bewußt`. No AI model writes like that, so leaving it in would give the detector another easy hint: "old spelling = human". We convert old spelling to the new rules. `ß` stays where it belongs after long vowels (`Maß`, `Fuß`, `heißt`).

2,364 samples come from documents where at least one word was converted. Usually it's just a few words (median: 2 per document). This counter was added partway through collection, so it's missing for 562 samples.

### Each document in only one split

All chunks of a document land in the same split (train / validation / test), decided by a hash of the document ID, so the model is never tested on a paper it saw in training. Result: 8,342 / 971 / 914 samples (81.6 / 9.5 / 8.9%).

### Other choices

- **At most 3 chunks per document**, spread evenly through the text, so the data covers many authors and topics. 2,483 documents gave 3 chunks, 1,140 gave 2, and 498 gave 1.
- **The first chunk of every document is skipped.** It usually holds the title, author names and abstract merged into the text.
- **Target lengths rotate** through 50, 150, 300, 600 and 1,000 words, starting at a different point for each document.
- **Every sample stores the ~120 words before it** (`context_before`, median 106 words). The AI side is generated from the same spot in the same paper.
- **The download was split between two people by year range** and merged afterwards. Duplicate chunks were removed by text hash.

## How to run

Open `collect_human_data.ipynb` and run the cells. Settings are in `human_data/config.py`; for a quick test, set `cfg.TARGET_PER_SOURCE = 30`. Output goes to `data/raw/human/`. Collection resumes where it stopped, so it can be interrupted at any time.

---

**Next:** [02 · AI data →](../02_ai-generated-data/)