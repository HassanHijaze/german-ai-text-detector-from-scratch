"""Quality filters for academic prose, plus old-to-new German spelling conversion."""

import re

from .pdf_text import KEEP_HYPHEN_NEXT_WORDS, glued_word_count
from .text_utils import GERMAN_ABBREVIATIONS, split_into_sentences, word_count


def is_good_prose(text):
    """
    Reject:
    - indexes
    - registers
    - OCR garbage
    - number-heavy chunks
    - tables/lists
    - fragments
    - extremely repetitive text

    Keep normal German prose.
    """

    text = text.strip()

    words = text.split()

    if len(words) < 35:
        return False

    # --------------------------------------------------------
    # Must finish as a complete sentence
    # --------------------------------------------------------

    if not re.search(r'[.!?]["\'»”’)\]}]*$', text):
        return False

    sentences = split_into_sentences(text)

    if not sentences:
        return False

    # --------------------------------------------------------
    # Numeric ratio
    # --------------------------------------------------------

    numeric_words = sum(1 for word in words if any(character.isdigit() for character in word))

    numeric_ratio = numeric_words / len(words)

    if numeric_ratio > 0.15:
        return False

    # --------------------------------------------------------
    # OCR-like mixed tokens:
    #
    # ro8
    # 2or
    # II6
    # r85
    # --------------------------------------------------------

    mixed_tokens = sum(
        1 for word in words if (re.search(r"[A-Za-zÄÖÜäöüß]", word) and re.search(r"\d", word))
    )

    if mixed_tokens / len(words) > 0.03:
        return False

    # --------------------------------------------------------
    # Mostly real alphabetic language
    # --------------------------------------------------------

    alphabetic_words = sum(1 for word in words if re.search(r"[A-Za-zÄÖÜäöüß]", word))

    if alphabetic_words / len(words) < 0.72:
        return False

    # --------------------------------------------------------
    # Index / register patterns
    #
    # Wissenschaft 50, 56, 66, 87
    # Zukunft 73, 108, 115, 257
    # --------------------------------------------------------

    index_patterns = re.findall(
        r"\b[A-Za-zÄÖÜäöüß\-]{3,}" r"\s+\d+" r"(?:\s*[,.;]\s*\d+){2,}", text
    )

    if len(index_patterns) >= 2:
        return False

    # --------------------------------------------------------
    # Too many number sequences separated by commas
    # --------------------------------------------------------

    numbered_sequences = re.findall(r"\d+\s*,\s*\d+\s*,\s*\d+", text)

    if len(numbered_sequences) >= 2:
        return False

    # --------------------------------------------------------
    # Too much punctuation compared with words
    # --------------------------------------------------------

    punctuation_count = len(re.findall(r"[,;:/|]", text))

    if punctuation_count / len(words) > 0.30:
        return False

    # --------------------------------------------------------
    # Sentence quality
    # --------------------------------------------------------

    sentence_lengths = [word_count(sentence) for sentence in sentences]

    very_short = sum(1 for length in sentence_lengths if length < 4)

    if len(sentence_lengths) >= 3 and very_short / len(sentence_lengths) > 0.40:
        return False

    # --------------------------------------------------------
    # Avoid text that is almost entirely repeated tokens
    # --------------------------------------------------------

    normalized_words = [re.sub(r"\W+", "", word.lower()) for word in words]

    normalized_words = [word for word in normalized_words if word]

    if normalized_words:
        unique_ratio = len(set(normalized_words)) / len(normalized_words)

        if unique_ratio < 0.22:
            return False

    return True


# Reference list entries:
#   "Müller, A. (2010): ..." / "Schmidt, K. H., ..."
#   "Luhmann, Niklas (1984): ..." / "(2007): Titel"
REFERENCE_ENTRY_RE = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüß\-]+,\s"
    r"(?:[A-ZÄÖÜ]\.\s?){1,3}"
    r"|\b[A-ZÄÖÜ][a-zäöüß\-]+,\s[A-ZÄÖÜ][a-zäöüß]+\s\(\d{4}"
    r"|\(\d{4}[a-z]?\)\s?:"
)

# A sentence that starts in lowercase after . ! ? usually means
# text fragments from tables / boxes were merged in:
# "... statt. durch methodische Ansätze."
LOWERCASE_AFTER_END_RE = re.compile(r"\b([A-Za-zÄÖÜäöüß]{3,})[.!?] ([a-zäöüß]{3,})")

# Old (pre-1996) spelling
OLD_SPELLING_RE = re.compile(
    r"\b(?:daß|muß|mußte|mußten|läßt|wußte|wußten|gewiß|"
    r"bißchen|Schluß|Prozeß|Mißbrauch|Einfluß|Anschluß)\b"
)

NEW_SPELLING_RE = re.compile(
    r"\b(?:dass|muss|musste|mussten|lässt|wusste|wussten|"
    r"gewiss|bisschen|Schluss|Prozess|Missbrauch|Einfluss|"
    r"Anschluss)\b"
)


# ------------------------------------------------------------
# Old -> new spelling (1996 reform), for 1990s texts.
# AI never writes "daß", so leaving it would be a shortcut.
#
# Old spelling wrote ß after a short vowel only at the end of a
# word or before a consonant ("daß", "läßt", "Prozeß",
# "bewußt", "häßlich"); the new spelling uses "ss" there.
# After a long vowel / diphthong ß stays ("Maß", "Fuß", "heißt").
# ------------------------------------------------------------

# Words / stems with a long vowel before ß -> keep ß
LONG_VOWEL_SZ = (
    "maß",
    "mäß",
    "spaß",
    "fraß",
    "saß",
    "gefäß",
    "gesäß",
    "fuß",
    "gruß",
    "buß",
    "straß",
)

SHORT_VOWEL_SZ_RE = re.compile(
    # single vowel (not part of ie / ei / au / eu / äu)
    # + ß at word end or before a consonant
    r"(?<![aeiouäöüAEIOUÄÖÜ])([aäeiuAÄEIU])ß"
    r"(?=[bcdfghjklmnpqrstvwxz]|\b)"
)


def modernize_spelling(text):
    """
    Returns (text, number_of_replacements).
        daß -> dass, läßt -> lässt, Prozeß -> Prozess,
        bewußt -> bewusst, mußte -> musste, Mißbrauch -> Missbrauch,
        müßte -> müsste, bißchen -> bisschen
    Keeps Maß, Maßnahme, gemäß, Fuß, Spaß, saß, heißt, groß.
    """

    count = 0

    def word_replace(match):
        nonlocal count

        word = match.group(0)
        lower = word.lower()

        if lower == "aß" or any(stem in lower for stem in LONG_VOWEL_SZ):
            return word

        new = SHORT_VOWEL_SZ_RE.sub(r"\1ss", word)

        # "Miß-" prefix before a vowel: "Mißerfolg", "mißachten"
        new = re.sub(r"^([Mm])iß", r"\1iss", new)

        # ü is ambiguous ("grüßt" long, "müßte" short): known words only
        new = re.sub(r"([MmKk])üß(t|te|ten|test|tet)\b", r"\1üss\2", new)

        new = new.replace("bißchen", "bisschen")

        if new != word:
            count += 1

        return new

    text = re.sub(r"[A-Za-zÄÖÜäöüß]*ß[A-Za-zÄÖÜäöüß]*", word_replace, text)

    return text, count


def uses_old_spelling(text):
    old = len(OLD_SPELLING_RE.findall(text))
    new = len(NEW_SPELLING_RE.findall(text))

    return old >= 2 and old > new


def is_good_academic_chunk(text):
    """
    v1 prose filter + academic-specific checks.
    Returns (ok, reason).
    """

    if not is_good_prose(text):
        return False, "chunk_not_prose"

    words = text.split()

    # Must start like a sentence, not mid-word / mid-sentence
    if not re.match(r"[\"'„“»(\[]?[A-ZÄÖÜ0-9]", text):
        return False, "chunk_starts_mid_sentence"

    # Book metadata (reviews): ISBN, prices
    if re.search(r"\bISBN\b|€\s?\d|\d,\d\d\s?€|\bEUR\s?\d", text):
        return False, "chunk_bibliographic_info"

    # Bibliography / reference list
    if len(REFERENCE_ENTRY_RE.findall(text)) >= max(2, len(words) * 0.01):
        return False, "chunk_reference_list"

    # Leftover hyphenation "Wirt- schaft"
    # (not "Säure- und Basenwerte")
    broken = [
        word
        for word in re.findall(r"[a-zäöüß]- ([a-zäöüß]+)", text)
        if word not in KEEP_HYPHEN_NEXT_WORDS
    ]

    if len(broken) > 1:
        return False, "chunk_broken_hyphenation"

    # PDF without spaces between words
    if glued_word_count(text) >= 2:
        return False, "chunk_missing_spaces"

    # Broken character encoding
    if "�" in text:
        return False, "chunk_broken_characters"

    # Leftover PDF-only characters (should be gone after cleaning)
    if re.search(r"[­​ﬀ-ﬄ]", text):
        return False, "chunk_pdf_characters"

    # Bullet lists: flattened into one line they read as
    # broken sentences ("... folgt: • durch ... • durch ...")
    if "•" in text:
        return False, "chunk_bullet_list"

    # Fragments from tables / boxes merged into the text
    if any(
        before.lower() + "." not in GERMAN_ABBREVIATIONS
        for before, _ in LOWERCASE_AFTER_END_RE.findall(text)
    ):
        return False, "chunk_broken_sentence"

    # Leftover URLs / e-mail / DOI
    if re.search(r"www\.|https?:|doi\.org|\S+@\S+\.\w+", text):
        return False, "chunk_url_or_email"

    return True, None
