"""Sentence-aware chunking of documents into samples of target lengths."""

import hashlib

from . import config as cfg
from .pdf_text import fix_spacing
from .text_utils import flatten_whitespace, split_into_sentences, word_count


def nearest_sentence_chunk(sentences, start_index, target_words):
    cumulative = 0

    previous_count = None
    previous_end = None

    for index in range(start_index, len(sentences)):
        cumulative += word_count(sentences[index])

        if cumulative >= target_words:
            current_distance = abs(cumulative - target_words)

            if previous_count is None:
                return (index + 1, cumulative)

            previous_distance = abs(previous_count - target_words)

            # exact tie -> choose longer chunk
            if current_distance <= previous_distance:
                return (index + 1, cumulative)

            return (previous_end, previous_count)

        previous_count = cumulative
        previous_end = index + 1

    return (len(sentences), cumulative)


def evenly_spaced(items, limit):
    if len(items) <= limit:
        return items

    if limit == 1:
        return [items[len(items) // 2]]

    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]

    return [items[index] for index in indices]


def chunk_document(text, document_id, max_chunks=None):
    if max_chunks is None:
        max_chunks = cfg.MAX_CHUNKS_PER_DOCUMENT
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    seed = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16)

    target_start = seed % len(cfg.TARGET_WORDS)

    candidates = []

    sentence_index = 0
    raw_chunk_number = 1

    while sentence_index < len(sentences):
        target = cfg.TARGET_WORDS[(target_start + raw_chunk_number - 1) % len(cfg.TARGET_WORDS)]

        end_index, actual = nearest_sentence_chunk(sentences, sentence_index, target)

        if end_index <= sentence_index:
            break

        # Too far from the target: usually the end of the
        # document or one huge "sentence" (table, list).
        within_limit = abs(actual - target) <= target * cfg.MAX_LENGTH_DEVIATION

        # fix_spacing: re-joining sentences can leave
        # "(S. 249ff.) . Auch" -> "(S. 249ff.). Auch"
        chunk = fix_spacing(flatten_whitespace(" ".join(sentences[sentence_index:end_index])))

        if chunk and within_limit:
            # Up to ~120 words of the text right before the chunk.
            # Lets the AI side be generated in the same context
            # (continue / rewrite this paper at this point).
            context = []

            for previous in reversed(sentences[:sentence_index]):
                if sum(word_count(s) for s in context) + word_count(previous) > 120 and context:
                    break

                context.insert(0, previous)

            candidates.append(
                {
                    "chunk_number": raw_chunk_number,
                    "target_length": target,
                    "text": chunk,
                    "context_before": fix_spacing(flatten_whitespace(" ".join(context))),
                }
            )

        sentence_index = end_index

        raw_chunk_number += 1

    # The first chunk usually holds author names, the title
    # and the abstract merged into the text: skip it.
    candidates = [candidate for candidate in candidates if candidate["chunk_number"] > 1]

    return evenly_spaced(candidates, max_chunks)
