"""Collect German academic human-written texts (1990–2022) from open repositories."""

from . import config
from .collect import collect_all, merge_datasets, summarize

__all__ = ["config", "collect_all", "merge_datasets", "summarize"]
