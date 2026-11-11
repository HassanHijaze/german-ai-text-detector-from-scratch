"""How much does each AI model change a text when it polishes it, and does that explain missed texts?

    python scripts/rewrite_similarity.py

Adapted from Ali Issa's rewrite-similarity analysis (his 09_model-comparison/model_comparison.ipynb,
section 4). AI-assisted test texts only:
    similarity        word-level similarity between each AI-assisted text and its human original
                      (found via seed_sample_id), with difflib.SequenceMatcher; 1.0 = unchanged
    by_generator      median similarity per generator
    by_similarity     mmBERT-base miss rate per similarity bin (missed = not predicted as AI-assisted)
    deepseek_by_bin   mmBERT-base miss rate of DeepSeek vs the other generators within each similarity
                      bin, with a Mantel-Haenszel test across the bins
    pedocs            within peDOCS: miss rate of DeepSeek vs each other generator, with counts and
                      Fisher's exact test, for all five transformers

IDs can be numeric (human 46, its AI rows with seed_sample_id 46) or prefixed (human-46, assisted-46).

Needs the dataset (texts and seed_sample_id) at data/hf-dataset-3class-full in the repository root,
so it runs separately from run.py. Prints the tables and saves results/missed-by-similarity.png.
"""

import difflib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact

from common import COLORS, FIG_DIR, GEN_NAMES, MODELS, REPO_ROOT, prediction_path
from plot import PERCENT, save, set_title

DATASET_DIR = REPO_ROOT / "data" / "hf-dataset-3class-full"
HUMAN, AI_ASSISTED = 0, 2
MAIN_MODEL = "mmbert-base"
DEEPSEEK = "deepseek-flash"

SIM_BINS = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
SIM_LABELS = ["≤60%", "60–70%", "70–80%", "80–90%", ">90%"]
PREFIXED_ID = re.compile(r"(?:human|assisted|generated)-(\d+)")


# ---------- Pure helpers ----------
def id_number(value) -> int:
    """The human original's number from a numeric ID (46, 46.0, "46") or a prefixed one ("human-46")."""
    if isinstance(value, str):
        text = value.strip()
        match = PREFIXED_ID.fullmatch(text)
        if match:
            return int(match.group(1))
        if text.isdigit():
            return int(text)
        raise ValueError(f"Unrecognised id: {value!r}")
    if value is None or pd.isna(value):
        raise ValueError("Missing id")
    if float(value) != int(value):
        raise ValueError(f"Not a whole-number id: {value!r}")
    return int(value)


def original_number(row_id, seed_sample_id) -> int:
    """Number of the human original of an AI row: seed_sample_id if set, otherwise a prefixed row id."""
    if seed_sample_id is not None and not pd.isna(seed_sample_id) and seed_sample_id != "":
        return id_number(seed_sample_id)
    if isinstance(row_id, str) and PREFIXED_ID.fullmatch(row_id.strip()):
        return id_number(row_id)
    raise ValueError(f"No human original for id {row_id!r}: seed_sample_id is empty")


def match_originals(assisted: pd.DataFrame, humans: pd.DataFrame) -> pd.Series:
    """Text of the human original for every row of `assisted` (index aligned), in either id format."""
    originals = {id_number(i): text for i, text in zip(humans["id"], humans["text"])}
    numbers = [original_number(i, s) for i, s in zip(assisted["id"], assisted["seed_sample_id"])]
    missing = [i for i, n in zip(assisted["id"], numbers) if n not in originals]
    if missing:
        raise ValueError(f"{len(missing)} AI-assisted texts have no human original, e.g. {missing[:5]}")
    return pd.Series([originals[n] for n in numbers], index=assisted.index)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def similarity(original: str, rewrite: str) -> float:
    """Word-level similarity (1.0 = unchanged), as in Ali Issa's version."""
    return difflib.SequenceMatcher(None, words(original), words(rewrite), autojunk=False).ratio()


def similarity_bin(values: pd.Series) -> pd.Series:
    """≤60%, 60–70%, 70–80%, 80–90%, >90%; each bin includes its upper edge (0.7 is in 60–70%)."""
    return pd.cut(values, SIM_BINS, labels=SIM_LABELS, include_lowest=True)


def mantel_haenszel(tables):
    """Common odds ratio and continuity-corrected test p-value over 2×2 tables [[a, b], [c, d]]."""
    num = den = deviation = variance = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        deviation += a - (a + b) * (a + c) / n
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    odds_ratio = num / den if den else np.inf
    statistic = (abs(deviation) - 0.5) ** 2 / variance if variance else 0.0
    return odds_ratio, float(chi2.sf(statistic, 1))


# ---------- Data ----------
def load_texts():
    """id, text, label, split, source, generator_model and seed_sample_id for every text in the dataset."""
    if not DATASET_DIR.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET_DIR}\n"
            "This analysis needs the texts and seed_sample_id from step 04. "
            "Put the dataset at data/hf-dataset-3class-full in the repository root and run it again."
        )
    from datasets import load_from_disk

    dataset = load_from_disk(str(DATASET_DIR))
    columns = ["id", "text", "label", "split", "source", "generator_model", "seed_sample_id"]
    return pd.concat([dataset[name].select_columns(columns).to_pandas() for name in dataset], ignore_index=True)


def build_table(texts):
    """One row per AI-assisted test text: similarity to its human original and missed flag per model."""
    assisted = texts[(texts["split"] == "test") & (texts["label"] == AI_ASSISTED)].copy()
    originals = match_originals(assisted, texts[texts["label"] == HUMAN])
    assisted["similarity"] = [similarity(o, t) for o, t in zip(originals, assisted["text"])]

    for key, name in MODELS.items():
        pred = pd.read_csv(prediction_path(key)).set_index("id")["pred"]
        if not assisted["id"].isin(pred.index).all():
            raise ValueError(f"{key}: predictions don't cover every AI-assisted test text")
        assisted[name] = assisted["id"].map(pred) != AI_ASSISTED   # True = missed
    return assisted.drop(columns="text").reset_index(drop=True)


# ---------- Analyses ----------
def by_generator(table):
    return (table.groupby("generator_model")["similarity"]
            .agg(texts="size", median="median")
            .rename(index=GEN_NAMES))


def by_similarity(table, model):
    return table.groupby(similarity_bin(table["similarity"]), observed=False)[model].agg(
        texts="size", missed="sum", miss_rate="mean")


def deepseek_by_bin(table, model):
    """Within each similarity bin: DeepSeek vs all other generators, plus a Mantel-Haenszel summary."""
    bins = similarity_bin(table["similarity"])
    is_deepseek = table["generator_model"] == DEEPSEEK
    rows, tables = [], []
    for label in SIM_LABELS:
        in_bin = bins == label
        ds, rest = table.loc[in_bin & is_deepseek, model], table.loc[in_bin & ~is_deepseek, model]
        rows.append({
            "bin": label,
            "deepseek_missed": int(ds.sum()), "deepseek_texts": len(ds),
            "deepseek_rate": ds.mean() if len(ds) else np.nan,
            "other_missed": int(rest.sum()), "other_texts": len(rest),
            "other_rate": rest.mean() if len(rest) else np.nan,
        })
        tables.append([[int(ds.sum()), int((~ds).sum())], [int(rest.sum()), int((~rest).sum())]])
    result = pd.DataFrame(rows)
    result["difference"] = result["deepseek_rate"] - result["other_rate"]

    odds_ratio, p = mantel_haenszel(tables)
    summary = {"mh_odds_ratio": odds_ratio, "mh_p": p}
    return result, summary


def pedocs_comparison(table):
    """Within peDOCS: DeepSeek vs each other generator, for every model."""
    pedocs = table[table["source"] == "pedocs"]
    others = [g for g in GEN_NAMES if g != DEEPSEEK and (pedocs["generator_model"] == g).any()]
    rows = []
    for name in MODELS.values():
        deepseek = pedocs.loc[pedocs["generator_model"] == DEEPSEEK, name]
        for other in others:
            rest = pedocs.loc[pedocs["generator_model"] == other, name]
            counts = [[int(deepseek.sum()), int((~deepseek).sum())], [int(rest.sum()), int((~rest).sum())]]
            odds_ratio, p = fisher_exact(counts)
            rows.append({
                "model": name, "vs": GEN_NAMES[other],
                "deepseek_missed": counts[0][0], "deepseek_texts": len(deepseek), "deepseek_rate": deepseek.mean(),
                "other_missed": counts[1][0], "other_texts": len(rest), "other_rate": rest.mean(),
                "odds_ratio": odds_ratio, "p_fisher": p,
            })
    return pd.DataFrame(rows)


# ---------- Figure ----------
def plot_missed_by_similarity(table, model):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(table))
    ax.bar(x, table["miss_rate"].fillna(0).values, 0.6, color=COLORS[model])
    for xi, (rate, n) in enumerate(zip(table["miss_rate"], table["texts"])):
        if n:
            ax.text(xi, rate + 0.005, f"{rate:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{label}\n(n = {n})" for label, n in zip(table.index, table["texts"])])
    ax.set_xlabel("Word-level similarity of the AI-assisted text to its human original")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_ylabel("Missed (not predicted as AI-assisted)")
    set_title(ax, f"AI-assisted texts missed by {model}, by rewrite similarity",
              "AI-assisted test texts only. Lighter rewrites keep more of the human original.")
    save(fig, "missed-by-similarity")


# ---------- Main ----------
def percent(value):
    return "–" if pd.isna(value) else f"{value:.1%}"


def main():
    texts = load_texts()
    table = build_table(texts)
    main_name = MODELS[MAIN_MODEL]
    print(f"{len(table):,} AI-assisted test texts, each with its human original")

    print("\n== Median similarity per generator")
    print(by_generator(table).to_string(float_format=lambda v: f"{v:.3f}"))

    sim = by_similarity(table, main_name)
    print(f"\n== {main_name} miss rate per similarity bin")
    print(sim.to_string(formatters={"miss_rate": percent}))

    per_bin, summary = deepseek_by_bin(table, main_name)
    print(f"\n== {main_name}: DeepSeek vs the other generators at the same similarity")
    shown = per_bin.copy()
    for col in ["deepseek_rate", "other_rate"]:
        shown[col] = shown[col].map(percent)
    shown["difference"] = shown["difference"].map(lambda v: "–" if pd.isna(v) else f"{v * 100:+.1f} pts")
    print(shown.to_string(index=False))
    print(f"Across the bins (Mantel-Haenszel): odds ratio {summary['mh_odds_ratio']:.2f}, p = {summary['mh_p']:.4f}")

    pedocs = pedocs_comparison(table)
    print("\n== Within peDOCS: DeepSeek vs each other generator (Fisher's exact test, two-sided)")
    shown = pedocs.copy()
    for col in ["deepseek_rate", "other_rate"]:
        shown[col] = shown[col].map(percent)
    shown["odds_ratio"] = shown["odds_ratio"].map(lambda v: f"{v:.2f}")
    shown["p_fisher"] = shown["p_fisher"].map(lambda v: f"{v:.4f}")
    print(shown.to_string(index=False))

    higher = pedocs.groupby("model", sort=False).apply(
        lambda d: bool((d["deepseek_rate"] > d["other_rate"]).all()), include_groups=False)
    print("\nDeepSeek's peDOCS AI-assisted texts missed more often than every other generator's:")
    for name, value in higher.items():
        print(f"  {name:18} {'yes' if value else 'no'}")
    print(f"  all five models:   {'yes' if higher.all() else 'no'}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figure to {FIG_DIR}")
    plot_missed_by_similarity(sim, main_name)
    return {"table": table, "by_generator": by_generator(table), "by_similarity": sim,
            "deepseek_by_bin": per_bin, "deepseek_summary": summary, "pedocs": pedocs}


if __name__ == "__main__":
    main()
