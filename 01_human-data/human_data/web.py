"""HTTP session with automatic retries."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)

    session.headers.update({"User-Agent": "German-Human-Text-Dataset/0.3 " "(research dataset)"})

    return session


session = create_session()


def get(url, *, params=None, headers=None, timeout=90):
    response = session.get(url, params=params, headers=headers, timeout=timeout)

    response.raise_for_status()

    return response
