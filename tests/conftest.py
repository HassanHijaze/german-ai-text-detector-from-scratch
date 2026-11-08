"""Shared helpers for the tests.

The stage folders start with numbers (07_fine-tuning, 09_model-comparison, 11_deployment),
so their code cannot be imported as normal packages. The fixtures below load it directly:

- 07 and 09 both contain a module called ``common``. Each folder is therefore imported
  on its own (its scripts folder first on sys.path) and its modules are removed from
  sys.modules again, so the two ``common`` modules never mix.
- 11_deployment/app.py loads a tokenizer and a model and builds a Gradio page when it is
  imported. ``transformers`` and ``gradio`` are replaced with small stubs while it loads,
  so no model is downloaded and no GPU or Gradio is needed.
"""

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # draw figures without opening windows

REPO_ROOT = Path(__file__).resolve().parents[1]
FINE_TUNING_SCRIPTS = REPO_ROOT / "07_fine-tuning" / "scripts"
MODEL_COMPARISON_SCRIPTS = REPO_ROOT / "09_model-comparison" / "scripts"
APP_PATH = REPO_ROOT / "11_deployment" / "app.py"


def load_stage_modules(script_dir: Path, names: list[str]) -> types.SimpleNamespace:
    """Import ``names`` from ``script_dir`` without leaking them into sys.modules."""
    clashing = set(names) | {"common"}
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name in clashing}
    sys.path.insert(0, str(script_dir))
    try:
        modules = {name: importlib.import_module(name) for name in names}
    finally:
        sys.path.remove(str(script_dir))
        for name in clashing:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return types.SimpleNamespace(**modules)


@pytest.fixture(scope="session")
def ft():
    """07_fine-tuning/scripts: common and evaluate."""
    return load_stage_modules(FINE_TUNING_SCRIPTS, ["common", "evaluate"])


@pytest.fixture(scope="session")
def mc():
    """09_model-comparison/scripts: common and analyze."""
    return load_stage_modules(MODEL_COMPARISON_SCRIPTS, ["common", "analyze"])


@pytest.fixture
def mc_full():
    """All 09_model-comparison scripts (common, analyze, plot, run), freshly imported for one test."""
    return load_stage_modules(MODEL_COMPARISON_SCRIPTS, ["common", "analyze", "plot", "run"])


@pytest.fixture
def ft_full():
    """07_fine-tuning scripts common, train and evaluate, freshly imported for one test."""
    return load_stage_modules(FINE_TUNING_SCRIPTS, ["common", "train", "evaluate"])


# ---------- Stubs for 11_deployment/app.py ----------
class StubTokenizer:
    """Counts words as tokens; enough for the parts of app.py that the tests use."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": list(range(len(texts.split())))}
        return {"input_ids": [list(range(len(t.split()))) for t in texts]}


class StubModel:
    """Returns equal logits for every input."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def eval(self):
        return self

    def __call__(self, input_ids=None, **kwargs):
        import torch

        batch = len(input_ids) if input_ids is not None else 1
        return types.SimpleNamespace(logits=torch.zeros(batch, 3))


class StubGradioError(Exception):
    pass


@pytest.fixture(scope="session")
def app():
    """11_deployment/app.py, imported with stubbed transformers and gradio."""
    stub_transformers = types.ModuleType("transformers")
    stub_transformers.AutoTokenizer = StubTokenizer
    stub_transformers.AutoModelForSequenceClassification = StubModel

    stub_gradio = MagicMock(name="gradio")
    stub_gradio.Error = StubGradioError

    saved = {name: sys.modules.get(name) for name in ("transformers", "gradio")}
    sys.modules["transformers"] = stub_transformers
    sys.modules["gradio"] = stub_gradio
    try:
        spec = importlib.util.spec_from_file_location("deployment_app", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module
