"""Prompts used to generate the AI texts.

The prompts are kept exactly as they were used for the dataset, so the
data can be reproduced. Two strategies exist for the AI-generated class:

- "source_based": the model sees the full human text and writes a new
  text on the same topic (used for the GPT and Gemini models)
- "topics": the model first extracts 3 short topics from the human text,
  then writes a new text from those topics only (used for DeepSeek)
"""

# ---- AI-assisted: rewrite the human text (same for every model) ----

ASSISTED_INSTRUCTIONS = """
Rewrite this German academic text so that it is clearly AI-assisted,
while still remaining based on the original human-written text.

You should:
- correct all grammar, spelling, and punctuation errors
- improve wording throughout the entire text
- rephrase most sentences to some degree
- improve sentence structure, fluency, and readability
- improve transitions and coherence between sentences
- replace awkward or imprecise expressions with more polished academic language
- reduce repetition and improve concision where appropriate
- make the overall style consistently professional and academically polished

But:
- preserve the original meaning and all important information
- preserve the overall argument and order of ideas
- preserve citations, quotations, numbers, names, and technical terminology
- keep the general paragraph structure recognizable
- do not completely rebuild the text from scratch
- do not introduce new factual information
- do not substantially change the topic or argument

The final text should show substantial AI-assisted editing throughout.
It should be clearly more than proofreading or minor correction,
but it should still remain recognizably derived from the original human text.

Return only the rewritten German text.
"""

# ---- AI-generated, strategy "source_based" ----

SOURCE_BASED_INSTRUCTIONS = """
Write a completely new German academic text based only on the topic and key ideas provided.

Do not edit, paraphrase, imitate, or preserve the writing style of any original text.
The supplied information is only a factual and thematic basis.

Act as the sole author of the new text.

Requirements:
- do not copy any numbers from the origin
- compose the entire text independently from scratch
- create completely new wording and sentence structures
- create your own logical structure and paragraph organization
- decide independently how to introduce, develop, connect, and conclude the ideas
- use highly polished, precise, professional academic German
- develop smooth transitions and a coherent argumentative flow
- freely reorganize and combine the supplied ideas
- remove unnecessary repetition
- preserve the underlying topic, facts, names, citations, numbers, and technical concepts
- keep the final length within ±20% of the target word count

Do not:

- do not quote or paraphrase original sentences
- extract only facts, concepts, and ideas
- do not include internal document references
- ignore references to tables, figures, pages, chapters, sections, appendices, or footnotes
- exclude expressions such as:
  "(vgl. Tabelle 19, S. 219)"
  "(siehe Abb. 3)"
  "(vgl. Kapitel 4)"
  "(S. 125)"
- preserve scholarly citations such as author names and publication years when they are relevant to the content

- reproduce phrases from the source
- follow the source sentence order
- paraphrase sentence by sentence
- imitate the source's style or syntax
- merely improve or polish existing formulations
- mention that a source text or key-point list was provided
- add unsupported factual claims

The final result must read as an independently authored academic text,
not as an edited or rewritten version of a human text.

Return only the final German academic text.
"""

# Filled with target_words, min_words, max_words, text
SOURCE_BASED_TEMPLATE = """
Target word count: approximately {target_words} words.

The final text should contain between approximately
{min_words} and {max_words} words.

Use the following human-written text only to understand:
- the topic
- the central ideas
- the arguments
- important facts
- names
- citations
- numbers
- technical terminology

Do not follow its wording or sentence structure.

SOURCE MATERIAL:

{text}
"""

# ---- AI-generated, strategy "topics" (two steps) ----
# Step 1: human text -> 3 short topics
TOPICS_EXTRACT_INSTRUCTIONS = """
Read the German academic text and identify exactly 3 main topics.

Rules:
- return exactly 3 topics
- each topic must contain no more than 10 words
- use only short topic phrases
- do not include full sentences
- do not include quotations
- do not include citations
- do not include numbers unless essential to the topic
- do not preserve wording from the original text unnecessarily
- do not include detailed arguments, claims, examples, or explanations
- do not include internal document references
- do not include tables, figures, pages, chapters, sections, appendices, or footnotes

The purpose is only to identify the broad subject areas of the text.

Return exactly this format:

1. ...
2. ...
3. ...
"""

# Step 2: topics -> new text
TOPICS_INSTRUCTIONS = """
Write a completely new German academic text from scratch based only on the
three supplied broad topics.

The three topics are only thematic guidance. Do not assume access to any
original source text.

Requirements:
- independently create the content
- independently create the arguments and explanations
- independently create every sentence
- independently determine the paragraph structure
- independently determine the order of ideas
- connect the three topics into one coherent academic text
- use natural, polished, professional academic German
- develop the topics with general academic reasoning
- keep the text focused on the supplied topics
- keep the final length within ±20% of the requested target word count

Do not:
- rewrite or paraphrase any source text
- imitate wording or sentence structure from another text
- mention an original text
- mention the topic extraction process
- mention the supplied topics as "notes"
- invent specific citations, authors, studies, statistics, or precise factual
  claims that were not supplied
- include internal references to tables, figures, pages, chapters, sections,
  appendices, or footnotes

The result should read like an independently authored academic text whose only
connection to another document is that it discusses the same three broad topics.

Return only the final German academic text.
"""

# Filled with target_words, min_words, max_words, notes
TOPICS_TEMPLATE = """
Write a new German academic text from the following content notes.

Target word count:
approximately {target_words} words

Allowed length:
{min_words} to {max_words} words

CONTENT NOTES:

{notes}
"""

# ---- Gemini: instructions and input are sent as one text ----
# Filled with instructions, text
GEMINI_INPUT_TEMPLATE = """
INSTRUCTIONS:

{instructions}

TASK INPUT:

{text}
"""
