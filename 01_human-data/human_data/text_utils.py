"""Basic text helpers: whitespace, counting, dates, language and sentence splitting."""

import datetime as dt
import hashlib
import re

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Make language detection deterministic
DetectorFactory.seed = 42


def clean_whitespace(text):
    if not text:
        return ""

    text = text.replace("\u00a0", " ")

    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r" *\n *", "\n", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def word_count(text):
    return len(re.findall(r"\S+", text))


def character_count(text):
    return len(text)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_date(value):
    if not value:
        return None

    value = str(value).strip()

    # YYYY-MM-DD
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", value)

    if match:
        try:
            return dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        except ValueError:
            pass

    # DD.MM.YYYY
    match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", value)

    if match:
        try:
            return dt.date(int(match.group(3)), int(match.group(2)), int(match.group(1)))

        except ValueError:
            pass

    # Year only
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", value)

    if match:
        return dt.date(int(match.group(1)), 12, 31)

    return None


def flatten_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


# ---- Language ----


def is_german(text):
    sample = text[:4000].strip()

    if len(sample) < 100:
        return False

    try:
        return detect(sample) == "de"

    except LangDetectException:
        return False


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None


# ---- Sentence detection ----

# Important:
# A "." is NOT always the end of a sentence.
# Compared lowercase, including the final period.

GERMAN_ABBREVIATIONS = {
    "z.b.",
    "u.a.",
    "d.h.",
    "i.d.r.",
    "o.ä.",
    "u.ä.",
    "z.t.",
    "u.u.",
    "i.e.",
    "e.g.",
    "bzw.",
    "usw.",
    "etc.",
    "ca.",
    "vgl.",
    "sog.",
    "ggf.",
    "ggfs.",
    "evtl.",
    "inkl.",
    "ggü.",
    "allg.",
    "entspr.",
    "urspr.",
    "beispielsw.",
    "insbes.",
    "zzgl.",
    "bzgl.",
    "insb.",
    "evtl.",
    "resp.",
    "vs.",
    "dr.",
    "prof.",
    "dipl.",
    "ing.",
    "nr.",
    "abs.",
    "art.",
    "str.",
    "bsp.",
    "bspw.",
    "geb.",
    "gest.",
    "hrsg.",
    "hg.",
    "mio.",
    "mrd.",
    "tsd.",
    "al.",
    "abb.",
    "tab.",
    "kap.",
    "aufl.",
    "bd.",
    "bde.",
    "jg.",
    "jh.",
    "jhd.",
    "ebd.",
    "ff.",
    "fn.",
    "anm.",
    "zit.",
    "ders.",
    "verf.",
    "sp.",
    "ziff.",
    "rn.",
    "rz.",
    "lit.",
    "univ.",
    "st.",
    "min.",
    "max.",
    "std.",
    # abbreviated first names: "Th. W. Adorno", "Ph. Müller"
    "th.",
    "ph.",
    "ch.",
    "fr.",
    "wm.",
    "jr.",
    "jan.",
    "feb.",
    "mär.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "sept.",
    "okt.",
    "nov.",
    "dez.",
}

# "im 19. Jahrhundert", "am 3. Mai", "die 2. Auflage":
# a number with a period followed by one of these words is an
# ordinal, not a sentence end. ("..., S. 45. Danach" stays a
# sentence end.)
ORDINAL_FOLLOWERS = {
    "januar",
    "februar",
    "märz",
    "april",
    "mai",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "dezember",
    "jahrhundert",
    "jahrhunderts",
    "jh",
    "jahrtausend",
    "auflage",
    "aufl",
    "kapitel",
    "band",
    "weltkrieg",
    "weltkriegs",
    "weltkrieges",
    "klasse",
    "jahrgang",
    "jahrgangsstufe",
    "lebensjahr",
    "semester",
    "quartal",
    "hälfte",
    "welle",
    "stufe",
    "grades",
    "ordnung",
    "sitzung",
    "wahlperiode",
    "legislaturperiode",
    "platz",
    "stelle",
    "schritt",
    "phase",
    "runde",
    "generation",
}

CLOSING_CHARACTERS = "\"'»”’)]}"


def token_before_position(text, position):
    start = position

    while start > 0:
        character = text[start - 1]

        if character.isspace() or character in '()[]{}"„“”«»':
            break

        start -= 1

    return text[start : position + 1]


def next_significant_character(text, position):
    i = position + 1

    while i < len(text) and text[i] in CLOSING_CHARACTERS:
        i += 1

    while i < len(text) and text[i].isspace():
        i += 1

    if i >= len(text):
        return None

    return text[i]


def is_sentence_end(text, position):
    character = text[position]

    if character not in ".!?":
        return False

    # For things such as "...", "?!", "!!"
    # only use the final punctuation character.

    if position + 1 < len(text) and text[position + 1] in ".!?":
        return False

    # "(... widerlegend?) zum ..." is not a sentence end
    if character in "!?":
        following = next_significant_character(text, position)

        return following is None or not following.islower()

    # Decimal number: 3.14

    if (
        position > 0
        and position + 1 < len(text)
        and text[position - 1].isdigit()
        and text[position + 1].isdigit()
    ):

        return False

    next_character = next_significant_character(text, position)

    # End of complete text
    if next_character is None:
        return True

    token = token_before_position(text, position).lower()

    if token in GERMAN_ABBREVIATIONS:
        return False

    # Single initial: A. Müller

    if re.fullmatch(r"[a-zäöüß]\.", token, flags=re.I):
        return False

    # z.B. / u.a.

    if re.fullmatch(r"(?:[a-zäöüß]\.){2,}", token, flags=re.I):
        return False

    # Ordinal: "im 19. Jahrhundert", "am 3. Mai"
    if re.fullmatch(r"\d{1,3}\.", token):
        following = re.match(r"[\s\"'»”’)\]}]*([A-Za-zÄÖÜäöüß]+)", text[position + 1 :])

        if following and following.group(1).lower() in ORDINAL_FOLLOWERS:
            return False

    # Usually continuation after abbreviation
    if next_character.islower():
        return False

    return True


def split_into_sentences(text):
    text = clean_whitespace(text)

    if not text:
        return []

    sentences = []

    start = 0
    i = 0

    while i < len(text):
        if is_sentence_end(text, i):
            end = i + 1

            while end < len(text) and text[end] in CLOSING_CHARACTERS:
                end += 1

            sentence = text[start:end].strip()

            if sentence:
                sentences.append(sentence)

            start = end

        i += 1

    # trailing unfinished text is intentionally ignored

    return sentences
