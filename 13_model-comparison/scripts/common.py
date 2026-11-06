"""Shared settings and data loading for the step 13 model comparison."""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Paths ----------
SCRIPTS_DIR = Path(__file__).resolve().parent   # 13_model-comparison/scripts
STEP_DIR = SCRIPTS_DIR.parent                     # 13_model-comparison
REPO_ROOT = STEP_DIR.parent
PRED_DIR = STEP_DIR / "predictions"
FIG_DIR = STEP_DIR / "results"

# ---------- Models ----------
# key -> display name
MODELS = {
    "gelectra-large": "GeLECTRA-large",
    "mmbert-base": "mmBERT-base",
    "mmbert-small": "mmBERT-small",
    "xlm-roberta-base": "XLM-RoBERTa base",
    "distilbert-german": "DistilBERT German",
}

# Okabe-Ito palette (colour-blind safe), one fixed colour per model
COLORS = {
    "GeLECTRA-large": "#0072B2",
    "mmBERT-base": "#009E73",
    "mmBERT-small": "#56B4E9",
    "XLM-RoBERTa base": "#E69F00",
    "DistilBERT German": "#CC79A7",
}

# ---------- Labels and groups ----------
LABELS = {0: "human", 1: "ai_generated", 2: "ai_assisted"}
LABEL_NAMES = {0: "Human", 1: "AI-generated", 2: "AI-assisted"}

GEN_NAMES = {
    "deepseek-flash": "DeepSeek V4 Flash",
    "gemini": "Gemini 3.8 Flash",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-6-luna": "GPT-6 Luna",
}
SRC_NAMES = {"ssoar": "SSOAR", "pedocs": "peDOCS", "refubium": "Refubium"}

LENGTH_BINS = [0, 70, 150, 300, 500, 750, np.inf]
LENGTH_LABELS = ["≤70", "71–150", "151–300", "301–500", "501–750", "751+"]

N_TEST = 2737
PROB_COLUMNS = ["p_human", "p_ai_generated", "p_ai_assisted"]


def prediction_path(key: str) -> Path:
    return PRED_DIR / f"{key}-test-predictions.csv"


def load_predictions():
    """Load all prediction files and check that they cover the same texts.

    Returns
    -------
    meta    : DataFrame with id, label, word_count, source, generator_model
    pred    : DataFrame of predicted labels, one column per model (display name)
    correct : DataFrame of booleans, one column per model
    conf    : DataFrame with the probability of the predicted class, one column per model
    """
    dfs = {key: pd.read_csv(prediction_path(key)) for key in MODELS}
    ref = next(iter(dfs.values()))

    for key, df in dfs.items():
        if len(df) != N_TEST:
            raise ValueError(f"{key}: expected {N_TEST} rows, found {len(df)}")
        if not (df["id"].values == ref["id"].values).all():
            raise ValueError(f"{key}: ids are not aligned with the other models")
        if not (df["label"].values == ref["label"].values).all():
            raise ValueError(f"{key}: labels are not aligned with the other models")

    meta = ref[["id", "label", "word_count", "source", "generator_model"]].copy()
    meta["generator_model"] = meta["generator_model"].fillna("none")
    y = meta["label"].values

    pred = pd.DataFrame({MODELS[k]: d["pred"].values for k, d in dfs.items()})
    correct = pd.DataFrame({name: pred[name].values == y for name in pred.columns})
    conf = pd.DataFrame({MODELS[k]: d[PROB_COLUMNS].max(axis=1).values for k, d in dfs.items()})
    return meta, pred, correct, conf


def length_bin(word_count: pd.Series) -> pd.Series:
    return pd.cut(word_count, bins=LENGTH_BINS, labels=LENGTH_LABELS)