"""Open-licence checks, OAI-PMH harvesting and PDF downloading."""

import xml.etree.ElementTree as ET
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from . import config as cfg
from .pdf_text import PdfSkipped, extract_pdf_text
from .text_utils import parse_date
from .web import get, session

#
# Only openly licensed texts (CC0, CC BY, CC BY-SA,
# public domain). NC / ND, "Deposit Licence" and
# "In Copyright" documents are skipped.
#
# SSOAR writes licences as text ("Creative Commons -
# Namensnennung"), Refubium / peDOCS as URLs.

NON_OPEN_LICENSE_MARKERS = (
    "by-nc",
    "by-nd",
    "noncommercial",
    "non-commercial",
    "nicht kommerz",
    "noderivatives",
    "no derivative",
    "keine bearbeitung",
    "deposit licence",
    "deposit license",
    "rightsstatements.org",
)


def detect_open_license(rights_values):
    if isinstance(rights_values, str):
        rights_values = [rights_values]

    combined = " ".join(rights_values).lower()

    if any(marker in combined for marker in NON_OPEN_LICENSE_MARKERS):
        return None

    if "publicdomain/zero" in combined or "public-domain/cc0" in combined or "cc0" in combined:
        return ("CC0 1.0", "https://creativecommons.org/" "publicdomain/zero/1.0/")

    if any(
        marker in combined
        for marker in (
            "licenses/by-sa/",
            "cc by-sa",
            "share alike",
            "sharealike",
            "weitergabe unter gleichen bedingungen",
        )
    ):
        return ("CC BY-SA", "https://creativecommons.org/" "licenses/by-sa/4.0/")

    if any(
        marker in combined
        for marker in (
            "licenses/by/",
            "cc by",
            "creative commons - attribution",
            "creative commons - namensnennung",
            "creative commons attribution",
            "creative commons namensnennung",
        )
    ):
        return ("CC BY", "https://creativecommons.org/" "licenses/by/4.0/")

    if "public domain" in combined or "gemeinfrei" in combined:
        return ("Public Domain", "https://creativecommons.org/" "publicdomain/mark/1.0/")

    return None


# ---- OAI-PMH harvesting ----

OAI_NS = "http://www.openarchives.org/" "OAI/2.0/"

DC_NS = "http://purl.org/dc/" "elements/1.1/"


def oai_pages(base_url, *, date_from=None, date_until=None):
    """
    Yields one list of records per OAI-PMH page.
    date_from / date_until filter on the record datestamp
    (when it was added / changed in the repository).
    """

    token = None

    while True:
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}

        else:
            params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}

            if date_from:
                params["from"] = date_from

            if date_until:
                params["until"] = date_until

        response = get(base_url, params=params)

        root = ET.fromstring(response.content)

        records = root.findall(f".//{{{OAI_NS}}}record")

        if records:
            yield records

        token_element = root.find(f".//{{{OAI_NS}}}resumptionToken")

        if token_element is None or not token_element.text or not token_element.text.strip():
            break

        token = token_element.text.strip()


def dc_values(record, field):
    return [
        element.text.strip()
        for element in record.findall(".//" f"{{{DC_NS}}}" f"{field}")
        if (element.text and element.text.strip())
    ]


def choose_publication_date(values):
    dates = [parse_date(value) for value in values]

    dates = [date for date in dates if date]

    return min(dates) if dates else None


def metadata_is_german(languages):
    valid = {"de", "deu", "ger", "german", "deutsch"}

    normalized = {value.lower().strip() for value in languages}

    return bool(normalized & valid)


# ---- PDF downloading ----


def find_pdf_url(landing_url):
    try:
        response = get(landing_url)

    except Exception:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    citation_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})

    if citation_pdf and citation_pdf.get("content"):
        return urljoin(landing_url, citation_pdf["content"])

    candidates = []

    for link in soup.find_all("a", href=True):
        href = link["href"]

        label = link.get_text(" ", strip=True).lower()

        lower_href = href.lower()

        if ".pdf" not in lower_href and "bitstream" not in lower_href:
            continue

        score = 0

        if ".pdf" in lower_href:
            score += 3

        if any(term in label for term in ["volltext", "download", "full text", "pdf"]):
            score += 3

        if any(term in lower_href for term in ["cover", "deckblatt", "titelblatt"]):
            score -= 10

        candidates.append((score, urljoin(landing_url, href)))

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][1]


def download_pdf_text(url):
    response = session.get(url, timeout=120, stream=True)

    response.raise_for_status()

    size = int(response.headers.get("Content-Length") or 0)

    if size > cfg.MAX_PDF_MB * 1024 * 1024:
        response.close()
        raise PdfSkipped("pdf_too_large")

    content = b""

    for part in response.iter_content(1024 * 256):
        content += part

        if len(content) > cfg.MAX_PDF_MB * 1024 * 1024:
            response.close()
            raise PdfSkipped("pdf_too_large")

    # e.g. an HTML landing / login page instead of the PDF
    if not content.lstrip().startswith(b"%PDF"):
        raise PdfSkipped("not_a_pdf")

    return extract_pdf_text(content)
