"""DIR (Directional expectation test): meaningful changes must move the probability in a known direction.

1. Replacing about half of the human text's sentences with the matching sentences of its
   AI-assisted version must raise P(AI-assisted).
2. Appending the AI-generated paragraph to the human text must raise
   P(AI-generated) + P(AI-assisted).
"""

import pytest

pytestmark = pytest.mark.behavioral

# Sentences 1, 3 and 5 of 6. Sentence 2 is identical in both versions, so it is not used.
REPLACED_SENTENCES = [0, 2, 4]


def test_half_ai_assisted_sentences_raise_p_ai_assisted(detector, demo_sentences, record):
    human, assisted = demo_sentences["human"], demo_sentences["ai_assisted"]
    assert len(human) == len(assisted), "the human and AI-assisted versions must have matching sentences"
    for i in REPLACED_SENTENCES:
        assert human[i] != assisted[i], f"sentence {i + 1} is the same in both versions"

    mixed = [assisted[i] if i in REPLACED_SENTENCES else sentence for i, sentence in enumerate(human)]
    before = detector.probabilities(" ".join(human))
    after = detector.probabilities(" ".join(mixed))
    record("DIR", f"human text with sentences {[i + 1 for i in REPLACED_SENTENCES]} from the AI-assisted version",
           human_text=before, mixed_text=after,
           p_ai_assisted_change=after["ai_assisted"] - before["ai_assisted"])

    assert after["ai_assisted"] > before["ai_assisted"], (
        f"P(AI-assisted) did not increase: {before['ai_assisted']:.6f} -> {after['ai_assisted']:.6f}"
    )


def test_appending_ai_generated_paragraph_raises_p_ai(detector, demo_texts, record):
    human = demo_texts["human"]
    combined = human + "\n\n" + demo_texts["ai_generated"]
    n_tokens = detector.num_tokens(combined)
    assert n_tokens <= detector.max_length, (
        f"combined text has {n_tokens} tokens; truncation at {detector.max_length} would hide the added paragraph"
    )

    before = detector.probabilities(human)
    after = detector.probabilities(combined)
    p_ai_before = before["ai_generated"] + before["ai_assisted"]
    p_ai_after = after["ai_generated"] + after["ai_assisted"]
    record("DIR", "human text + AI-generated paragraph", tokens=n_tokens,
           human_text=before, combined_text=after,
           p_ai_before=round(p_ai_before, 6), p_ai_after=round(p_ai_after, 6))

    assert p_ai_after > p_ai_before, f"P(AI-generated) + P(AI-assisted) did not increase: {p_ai_before:.6f} -> {p_ai_after:.6f}"
