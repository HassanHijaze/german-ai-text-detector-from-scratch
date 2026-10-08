"""Generate AI-assisted and AI-generated German academic texts from the human data."""

from . import config
from .generate import find_missing, run, show_example

__all__ = ["config", "run", "find_missing", "show_example"]
