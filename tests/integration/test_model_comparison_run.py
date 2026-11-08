"""Full run of 09_model-comparison/scripts/run.py on small fake prediction files."""

import numpy as np
import pandas as pd

N_ROWS = 300
GENERATORS = ["deepseek-flash", "gemini", "gpt-5.6-luna", "gpt-6-luna"]
SOURCES = ["ssoar", "pedocs", "refubium"]
FIGURES = ["mcnemar-heatmap", "agreement-summary", "error-by-generator", "error-by-source",
           "confidence-correct-vs-wrong", "hardest-texts-by-length"]


def fake_predictions(model_index, rng):
    """One prediction file: same texts for every model, different (deliberate) errors."""
    ids = np.arange(1, N_ROWS + 1)
    labels = ids % 3
    generator = np.where(labels == 0, None, np.array(GENERATORS)[ids % 4])
    wrong = rng.random(N_ROWS) < 0.04 + 0.03 * model_index  # each model makes more errors than the last
    wrong |= ids % 37 == 0                                   # texts that every model gets wrong
    pred = np.where(wrong, (labels + 1 + (ids % 2)) % 3, labels)
    probs = np.full((N_ROWS, 3), 0.05)
    probs[np.arange(N_ROWS), pred] = 0.90
    return pd.DataFrame({
        "id": ids, "label": labels, "word_count": rng.integers(20, 900, N_ROWS),
        "source": np.array(SOURCES)[(ids // 3) % 3],  # independent of the label (ids % 3)
        "generator_model": generator, "pred": pred,
        "p_human": probs[:, 0], "p_ai_generated": probs[:, 1], "p_ai_assisted": probs[:, 2],
    })


def test_full_run_creates_all_figures(mc_full, tmp_path, monkeypatch):
    pred_dir, fig_dir = tmp_path / "predictions", tmp_path / "results"
    pred_dir.mkdir()
    rng = np.random.default_rng(0)
    for i, key in enumerate(mc_full.common.MODELS):
        fake_predictions(i, rng).to_csv(pred_dir / f"{key}-test-predictions.csv", index=False)

    monkeypatch.setattr(mc_full.common, "PRED_DIR", pred_dir)
    monkeypatch.setattr(mc_full.common, "FIG_DIR", fig_dir)
    monkeypatch.setattr(mc_full.common, "N_TEST", N_ROWS)
    # plot.py imports FIG_DIR by name ("from common import FIG_DIR"), so it needs its own patch
    monkeypatch.setattr(mc_full.plot, "FIG_DIR", fig_dir)

    mc_full.run.main()

    for name in FIGURES:
        figure = fig_dir / f"{name}.png"
        assert figure.exists(), f"{figure.name} was not created"
        assert figure.stat().st_size > 1000, f"{figure.name} is (nearly) empty"
    assert sorted(p.name for p in fig_dir.iterdir()) == sorted(f"{n}.png" for n in FIGURES)
