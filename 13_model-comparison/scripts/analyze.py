"""Compute the comparison statistics from the saved predictions.

Nothing is written to disk: main() prints the results and returns them for plot.py.
    mcnemar           exact McNemar test for every pair of models (Holm-corrected)
    summary           texts all models get right / wrong / disagree on
    by_generator      error rate per generator (AI texts only)
    by_source         error rate per source (all texts)
    confidence        confidence of correct vs wrong predictions
    hardest           texts that every model misclassifies
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import LABELS, load_predictions


def holm(p_values):
    """Holm-Bonferroni correction."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def mcnemar_table(correct):
    rows = []
    for a, b in combinations(correct.columns, 2):
        a_only = int((correct[a] & ~correct[b]).sum())
        b_only = int((~correct[a] & correct[b]).sum())
        n = a_only + b_only
        p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
        rows.append({"model_a": a, "model_b": b,
                     "a_right_b_wrong": a_only, "a_wrong_b_right": b_only, "p_value": p})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_value"])
    table["significant"] = table["p_holm"] < 0.05
    better = np.where(table["a_right_b_wrong"] > table["a_wrong_b_right"],
                      table["model_a"] + " better", table["model_b"] + " better")
    table["result"] = np.where(table["significant"], better, "no significant difference")
    return table


def agreement_summary(correct):
    all_correct = correct.all(axis=1)
    all_wrong = (~correct).all(axis=1)
    summary = pd.DataFrame([{
        "texts": len(correct),
        "all_models_correct": int(all_correct.sum()),
        "all_models_wrong": int(all_wrong.sum()),
        "models_disagree": int((~all_correct & ~all_wrong).sum()),
    }])
    return summary


def error_by_group(correct, groups, mask=None):
    if mask is None:
        mask = pd.Series(True, index=correct.index)
    errors = ~correct[mask]
    table = pd.DataFrame({name: errors[name].groupby(groups[mask]).mean() for name in correct.columns})
    table.insert(0, "samples", groups[mask].value_counts())
    table.index.name = groups.name
    return table


def confidence_table(correct, conf):
    rows = []
    for name in correct.columns:
        right, wrong = conf[name][correct[name]], conf[name][~correct[name]]
        rows.append({
            "model": name,
            "mean_conf_correct": right.mean(),
            "mean_conf_wrong": wrong.mean(),
            "median_conf_wrong": wrong.median(),
            "wrong_with_conf_over_90": (wrong >= 0.9).mean(),
        })
    return pd.DataFrame(rows)


def hardest_texts(meta, pred, correct):
    all_wrong = (~correct).all(axis=1)
    hard = meta.loc[all_wrong, ["id", "label"]].copy()
    hard["true_label"] = hard.pop("label").map(LABELS)
    for name in pred.columns:
        hard[name] = pred.loc[all_wrong, name].map(LABELS).values
    for col in ["word_count", "source", "generator_model"]:
        hard[col] = meta.loc[all_wrong, col].values
    return hard.sort_values("word_count")


def main():
    meta, pred, correct, conf = load_predictions()
    results = {
        "meta": meta,
        "correct": correct,
        "names": list(pred.columns),
        "mcnemar": mcnemar_table(correct),
        "summary": agreement_summary(correct),
        "by_generator": error_by_group(correct, meta["generator_model"], mask=meta["label"] != 0),
        "by_source": error_by_group(correct, meta["source"]),
        "confidence": confidence_table(correct, conf),
        "hardest": hardest_texts(meta, pred, correct),
    }

    print(results["mcnemar"][["model_a", "model_b", "a_right_b_wrong", "a_wrong_b_right", "p_holm", "result"]], "\n")
    print(results["summary"], "\n")
    print(results["by_generator"], "\n")
    print(results["by_source"], "\n")
    print(results["confidence"], "\n")
    print(f"{len(results['hardest'])} texts are misclassified by all models")
    return results


if __name__ == "__main__":
    main()