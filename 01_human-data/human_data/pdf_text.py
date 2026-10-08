"""PDF text extraction and cleaning.

Uses font sizes to keep only the main body text (drops footnotes,
headings and captions), then repairs common PDF problems.
"""

import re
from collections import Counter

import pymupdf

from . import config as cfg
from .text_utils import clean_whitespace, detect_language, word_count

#
# Uses font sizes to keep only the main body text:
#   - footnotes, captions, superscript footnote markers
#     (smaller than body text) are dropped
#   - titles and headings (larger) are dropped
#   - cover pages, tables of contents, running headers,
#     English paragraphs and everything after the reference
#     list heading are dropped
#   - words hyphenated across line breaks are re-joined

BACKMATTER_HEADING_RE = re.compile(
    r"^(?:[\dIVX]+(?:\.\d+)*\.?\s*)?"
    r"(?:Literatur(?:verzeichnis)?|Literatur-\s?und\s?Quellenverzeichnis|"
    r"Quellen(?:verzeichnis)?|Quellen-\s?und\s?Literaturverzeichnis|"
    r"Bibliogra(?:ph|f)ie|References|Bibliography|"
    r"Verwendete\s+Literatur|Zitierte\s+Literatur|"
    r"Anhang|Anhänge|Appendix|Anmerkungen|Endnoten|"
    r"Register|Sachregister|Personenregister|Index)\s*:?$",
    flags=re.I,
)

COVER_PAGE_MARKERS = (
    "nutzungsbedingungen",
    "terms of use",
    "empfohlene zitierung",
    "suggested citation",
    "zur verfügung gestellt in kooperation mit",
)

# "Einleitung ........ 5"   /   "2.1 Methode . . . . 12"
TOC_LINE_RE = re.compile(r"(?:\.\s?){4,}\s*\d+\s*$" r"|…+\s*\d+\s*$")

# Table-of-contents entry without dot leaders:
# "2.4 Kognitive Kompetenzen von Kindern im Schulalter 15"
TOC_ENTRY_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+\D{3,120}?\s\d{1,4}$")

# Compound hyphen that must stay: "Ein- und Ausgang"
KEEP_HYPHEN_NEXT_WORDS = {"und", "oder", "bzw", "bzw.", "sowie", "bis", "als"}


def join_lines(lines):
    """
    Join the lines of one PDF text block into a paragraph,
    re-joining words hyphenated across the line break:
        "Wirt-" + "schaft"  -> "Wirtschaft"
        "Ein-"  + "und ..." -> "Ein- und ..."
    """

    text = ""

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if not text:
            text = line
            continue

        next_word = line.split()[0]

        if (
            re.search(r"[A-Za-zÄÖÜäöüß]-$", text)
            and next_word[0].islower()
            and next_word.lower().rstrip(",.;:") not in KEEP_HYPHEN_NEXT_WORDS
        ):
            text = text[:-1] + line
        else:
            text = text + " " + line

    return text


def line_text(spans):
    """
    Join the spans of one PDF line. Some PDFs position words
    instead of writing a space; add the space back when there
    is a visible gap ("reguläreRezension" -> "reguläre Rezension").
    """

    text = ""
    previous_x1 = None

    for piece, size, _, x0, x1 in spans:
        if (
            text
            and previous_x1 is not None
            and x0 - previous_x1 > size * 0.15
            and not text[-1].isspace()
            and not piece[0].isspace()
        ):
            text += " "

        text += piece
        previous_x1 = x1

    return text


def fix_hyphenation(text):
    """
    "Wirt- schaft" -> "Wirtschaft",
    but keep "Berufs- und Wirtschaftspädagogik".
    """

    def join(match):
        if match.group(2).lower() in KEEP_HYPHEN_NEXT_WORDS:
            return match.group(0)

        return match.group(1) + match.group(2)

    text = re.sub(r"([A-Za-zÄÖÜäöüß]{2,})- ([a-zäöüß]+)", join, text)

    # "Covid- 19" -> "Covid-19", "Online- Plattformen" -> "Online-Plattformen"
    return re.sub(r"([A-Za-zÄÖÜäöüß]{2,})- (\d+\b|[A-ZÄÖÜ][a-zäöüß])", r"\1-\2", text)


LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalize_characters(text):
    """
    Invisible / PDF-only characters a human would never type
    (and an AI would never produce):
        "Infrastruk\\u00ad turen" -> "Infrastrukturen"   (soft hyphen)
        "\\ufb01" -> "fi"                                (ligature)
    """

    text = re.sub(r"­\s*", "", text)

    text = re.sub(r"[​‌‍⁠﻿]", "", text)

    # Control characters ("\x07")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    text = re.sub(r"[‐‑]", "-", text)

    for ligature, letters in LIGATURES.items():
        text = text.replace(ligature, letters)

    # Bullet glyphs (incl. private-use Symbol-font bullets
    # like U+F0B7) -> "•", so list chunks can be detected
    text = re.sub(r"[-•▪●◦‣⁃■□]", "•", text)

    return text


def remove_footnote_numbers(text):
    """
    Footnote numbers printed in body font size:
        "gearbeitet30."         -> "gearbeitet."
        "(Storytelling)29 dar"  -> "(Storytelling) dar"
        "werden.12 Die"         -> "werden. Die"
    Keeps "COVID-19", "CO2", "Abb.1", "[15]", "S. 45".
    """

    text = re.sub(r"(?<=[a-zäöüß)”“])\d{1,3}(?=[\s.,;:!?)]|$)", "", text)

    return re.sub(r"(?<=[a-zäöüß][.,;:!?])\d{1,3}(?=\s+[A-ZÄÖÜ„\"])", "", text)


def fix_spacing(text):
    """
    "(s.o. II. 3.) ."  -> "(s.o. II. 3.)."
    "( Abb. 2)"        -> "(Abb. 2)"
    """

    text = re.sub(r"\s+([,.;:!?])(?=\s|$)", r"\1", text)

    text = re.sub(r"([(\[„])\s+", r"\1", text)

    # Closing German quote after a line break: "… sind. “ Ein" -> "… sind.“ Ein"
    text = re.sub(r"([.!?,;:])\s+“(?=\s|$)", r"\1“", text)

    return re.sub(r"\s+([)\]])", r"\1", text)


def clean_pdf_paragraph(text):
    return fix_spacing(remove_footnote_numbers(fix_hyphenation(normalize_characters(text))))


# Words glued together by a PDF without spaces:
#   "ImgesamtenProzesswerden", "beiGesundheitsfachkräftenwährend",
#   "Erkrankungaufweist,sondernbereitsvor"
# but not brand names ("YouTube", "PowerPoint", "ResearchGate"),
# gender forms ("AssistentInnen") or long real compounds
# ("kommunikationswissenschaftlichen").
CAMEL_CASE_RE = re.compile(r"[a-zäöüß][A-ZÄÖÜ]")

COMMA_GLUED_RE = re.compile(r"[a-zäöüß]{2},[a-zäöüß]{2}")


def glued_word_count(text):
    count = 0

    for word in text.split():
        core = word.strip(".,;:!?()[]{}\"'„“”»«‘’–-")

        if COMMA_GLUED_RE.search(core):
            count += 1

        elif len(core) >= 18 and CAMEL_CASE_RE.search(core.replace("Innen", "innen")):
            count += 1

    return count


def document_damage(text):
    """
    Whole-document checks: if the PDF lost its spaces or its
    character encoding, the damage is also in places no
    chunk filter can see ("Eingriffwird", "Leh renden").
    A few "\\ufffd" in a formula are fine; hundreds are not.
    """

    broken = text.count("�")

    if broken >= 5 and broken / max(1, word_count(text)) > 0.0005:
        return "document_broken_characters"

    glued = glued_word_count(text)

    if glued >= 5 and glued / max(1, word_count(text)) > 0.002:
        return "document_missing_spaces"

    return None


def pdf_blocks(document):
    """
    Returns a list of pages; each page is a list of blocks;
    each block is a list of lines; each line is a list of
    (text, size, bold, x0, x1) spans.
    """

    pages = []

    for page in document:
        data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

        blocks = []

        for block in data["blocks"]:
            if block.get("type") != 0:
                continue

            lines = []

            for line in block["lines"]:
                spans = [
                    (
                        span["text"],
                        round(span["size"] * 2) / 2,
                        bool(span["flags"] & 16),
                        span["bbox"][0],
                        span["bbox"][2],
                    )
                    for span in line["spans"]
                    if span["text"].strip()
                ]

                if spans:
                    lines.append(spans)

            if lines:
                blocks.append(lines)

        pages.append(blocks)

    return pages


def body_font_size(pages):
    sizes = Counter()

    for blocks in pages:
        for lines in blocks:
            for spans in lines:
                for text, size, *_ in spans:
                    sizes[size] += len(text)

    if not sizes:
        return None

    return sizes.most_common(1)[0][0]


class PdfSkipped(Exception):
    """A PDF that yields no usable text, with the reason."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def extract_pdf_text(pdf_bytes):
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise PdfSkipped("pdf_unreadable")

    try:
        if document.needs_pass:
            raise PdfSkipped("pdf_encrypted")

        if document.page_count > cfg.MAX_PDF_PAGES:
            raise PdfSkipped("pdf_too_many_pages")

        pages = pdf_blocks(document)

    except PdfSkipped:
        raise

    except Exception:
        raise PdfSkipped("pdf_unreadable")

    finally:
        document.close()

    body_size = body_font_size(pages)

    # Scanned PDF: images only, no text layer
    if (
        body_size is None
        or sum(
            len(span[0])
            for blocks in pages
            for lines in blocks
            for spans in lines
            for span in spans
        )
        < 200 * max(1, len(pages)) * 0.1
    ):
        raise PdfSkipped("pdf_no_text_layer")

    # Lines repeated on many pages = running headers / footers
    line_counts = Counter()

    for blocks in pages:
        page_lines = {line_text(spans).strip().lower() for lines in blocks for spans in lines}

        line_counts.update(page_lines)

    repeated = set()

    if len(pages) >= 5:
        threshold = max(3, int(len(pages) * 0.25))

        repeated = {
            line for line, count in line_counts.items() if count >= threshold and len(line) <= 120
        }

    paragraphs = []

    for page_number, blocks in enumerate(pages):
        page_text = " ".join(line_text(spans) for lines in blocks for spans in lines).lower()

        # Repository cover sheet (SSOAR, peDOCS, ...)
        if page_number < 2 and sum(marker in page_text for marker in COVER_PAGE_MARKERS) >= 2:
            continue

        stop = False

        for lines in blocks:
            full_lines = [line_text(spans).strip() for spans in lines]

            full_text = " ".join(full_lines).strip()

            # Reference list / appendix starts:
            # ignore the rest of the document
            if page_number >= len(pages) * 0.4 and BACKMATTER_HEADING_RE.match(full_text):
                stop = True
                break

            # Table of contents lines / index-like blocks
            if any(TOC_LINE_RE.search(line) for line in full_lines):
                continue

            if (
                len(full_lines) >= 3
                and sum(bool(re.search(r"\d+\s*$", line)) for line in full_lines) / len(full_lines)
                >= 0.5
            ):
                continue

            # Keep only spans in the body font size:
            # drops footnotes, superscripts, captions, headings
            body_lines = []

            for spans, full_line in zip(lines, full_lines):
                if full_line.lower() in repeated:
                    continue

                body_spans = [span for span in spans if abs(span[1] - body_size) <= 0.5]

                if not body_spans:
                    continue

                kept = line_text(body_spans).strip()

                # Bold line without final punctuation:
                # a heading in body font size
                if all(span[2] for span in body_spans) and not re.search(r"[.!?:]$", kept):
                    continue

                if kept:
                    body_lines.append(kept)

            paragraph = clean_pdf_paragraph(
                clean_whitespace(join_lines(body_lines)).replace("\n", " ")
            )

            # Paragraph numbers "[36] ..." / "(12) ..."
            paragraph = re.sub(r"^[\[(]\d{1,3}[\])]\s+", "", paragraph)

            if not paragraph:
                continue

            # Keyword lines
            if re.match(
                r"(?:Schlagw[oö]rte?|Stichw[oö]rte?r?|" r"Keywords?|Key words|JEL)\s*:",
                paragraph,
                flags=re.I,
            ):
                continue

            words = paragraph.split()

            # Headings in body font, page numbers, stray lines
            if len(words) < 8 and not re.search(r"[.!?:]$", paragraph):
                continue

            # English abstract, quotes in other languages
            if len(words) >= 25:
                language = detect_language(paragraph)

                if language and language != "de":
                    continue

            # Paragraph continues after a page / column break:
            # "... Anträge für Arbeitslosen-" + "geld und ..."
            if (
                paragraphs
                and not re.search(r"[.!?:;\"“”»)]$", paragraphs[-1])
                and paragraph[0].islower()
            ):
                paragraphs[-1] = clean_pdf_paragraph(join_lines([paragraphs[-1], paragraph]))
                continue

            paragraphs.append(paragraph)

        if stop:
            break

    # Short lines that are not sentences: headings, signatures
    # ("Ministerin für Bildung, ... des Landes Schleswig-Holstein"),
    # table-of-contents entries ("2.3 Übergänge im Sekundarbereich I 11")
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not (len(paragraph.split()) < 15 and not re.search(r"[.!?:;\"“”»)]$", paragraph))
        and not TOC_ENTRY_RE.match(paragraph)
    ]

    text = clean_plain_text("\n\n".join(paragraphs))

    if not text:
        raise PdfSkipped("pdf_no_body_text")

    return text


def clean_plain_text(text):
    text = clean_whitespace(text)

    # URLs
    text = re.sub(r"https?://\S+", " ", text)

    # isolated page numbers
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)

    # lines consisting mainly of symbols
    text = re.sub(r"(?m)^[=_\-*•.]{5,}$", "", text)

    return clean_whitespace(text)
