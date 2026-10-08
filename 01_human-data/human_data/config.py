"""Settings for collecting the human (German academic) texts.

Change values here, or override them from the notebook:
    from human_data import config as cfg
    cfg.TARGET_PER_SOURCE = 30
"""

import datetime as dt
import random
from pathlib import Path

# Sources: SSOAR, Refubium (FU Berlin) and peDOCS (DIPF), open licences only.

# Publication date range. Never set the end later than 2022-12-31
# (public LLM tools came out after that). The download can be split
# between people by year range and combined with merge_datasets().
MIN_PUBLICATION_DATE = dt.date(1990, 1, 1)
CUTOFF_DATE = dt.date(2022, 12, 31)

# Repo root (this file is in <repo>/01_human_data/human_data/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Output: meta.json + human/<id>.txt
DATA_DIR = REPO_ROOT / "data" / "raw" / "human"

# Chunks per source.
# Small test run: 30 (about 10 documents per source)
TARGET_PER_SOURCE = 6000

TARGET_WORDS = (50, 150, 300, 600, 1000)

# A chunk must end on a sentence boundary and stay within
# ±30% of its target (a 50-word target accepts 35–65 words).
MAX_LENGTH_DEVIATION = 0.30

# Few chunks per document -> many different documents,
# authors and topics.
MAX_CHUNKS_PER_DOCUMENT = 3

# Minimum clean body text per document (after removing
# cover pages, footnotes, references, English parts, ...)
MIN_DOCUMENT_WORDS = 400

# Skip very large PDFs. Dissertations with many images are
# often 40–100 MB, and they are the most thesis-like texts.
MAX_PDF_MB = 100
MAX_PDF_PAGES = 400

# Spread documents through each repository instead of
# taking consecutive records (often the same book series).
# These only decide the ORDER: the first pass takes a few
# documents per page / year window, a second pass then uses
# every remaining eligible document until the target is reached.
MAX_DOCS_PER_OAI_PAGE = 2
MAX_DOCS_PER_YEAR_WINDOW = 4

# Document-level train / validation / test split.
# All chunks of one document always land in the same split,
# so the model is never tested on a document it trained on.
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}

RANDOM = random.Random(42)


# ---- Endpoints ----

SSOAR_OAI = "https://www.ssoar.info/OAIHandler/request"

# Note: "/oai/dini" returns HTTP 400, "/oai/request" works.
REFUBIUM_OAI = "https://refubium.fu-berlin.de/oai/request"

PEDOCS_OAI = "https://www.pedocs.de/oai2/oai2.php"


SOURCE_INFO = {
    "ssoar": {
        "name": "SSOAR",
        "source_type": "academic_social_sciences",
        "oai": SSOAR_OAI,
        "landing_host": "ssoar.info",
        "first_year": 2012,
    },
    "refubium": {
        "name": "Refubium (FU Berlin)",
        "source_type": "academic_university",
        "oai": REFUBIUM_OAI,
        "landing_host": "refubium.fu-berlin.de",
        "first_year": 1998,
    },
    "pedocs": {
        "name": "peDOCS (DIPF)",
        "source_type": "academic_education",
        "oai": PEDOCS_OAI,
        "landing_host": "pedocs.de",
        "first_year": 1999,
    },
}
