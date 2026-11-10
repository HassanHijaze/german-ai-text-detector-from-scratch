"""Evaluate one fine-tuned model on train / validation / test.

    python scripts/evaluate.py mmbert-base              # model from 07_fine-tuning/artifacts/mmbert-base-3class/
    python scripts/evaluate.py mmbert-base --from-hub   # model from Hugging Face (HassanHijazi/german-ai-text-<name>)

(run from 07_fine-tuning/)

Same procedure as the original notebooks (archive/): a calibration temperature is
fitted on the validation split, and all metrics use the temperature-scaled
probabilities. Writes to 07_fine-tuning/results/<name>/new-run/, so the reported
results in results/<name>/ are never overwritten:

    <name>-metrics.csv                 accuracy, macro-F1, per-class errors, Brier, log loss per split
    <name>-calibration.csv             validation Brier score and log loss before / after scaling
    <name>-3-class-confusion-matrix.csv / <name>-confmat-3class.png
    <name>-confmat-human-ai.csv / .png  Human vs AI-generated (AI-assisted texts left out)
    <name>-length-bias-3class.csv      test results per text-length bin

detector-config.json (same keys as the notebooks) is saved next to the model:
in the artifacts folder for a local model, in results/<name>/new-run/ for --from-hub.
"""

import argparse
import json
import platform
from importlib.metadata import version

import common

LABELS = [0, 1, 2]

# Word-count bins for the length-bias table: (lower, upper, label), both ends inclusive.
LENGTH_BINS = [
    (0, 70, "≤70"),
    (71, 150, "71–150"),
    (151, 300, "151–300"),
    (301, 500, "301–500"),
    (501, 750, "501–750"),
    (751, None, "751+"),
]


# ---------- Calibration (identical to the notebooks) ----------
# Temperature scaling, fitted on the validation split (LBFGS on log T).
def fit_temperature(logits, labels):
    import torch

    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_tensor / log_temperature.exp(), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach())


def probabilities_from_logits(logits, temperature=1.0):
    import numpy as np

    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def multiclass_brier_score(y_true, probabilities, n_classes=3):
    import numpy as np

    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))


# ---------- Prediction ----------
def predict_logits(model, tokenizer, tokenized_split, work_dir):
    import numpy as np
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=str(work_dir),
        per_device_eval_batch_size=common.CONFIG["training"]["per_device_eval_batch_size"],
        report_to="none",
        **common.precision_flags(),
    )
    trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                      data_collator=common.data_collator(tokenizer))
    return np.asarray(trainer.predict(tokenized_split).predictions)


def predict_probabilities(model, tokenizer, texts, temperature, work_dir):
    """Calibrated probabilities for a list of texts, with the same steps as evaluate():
    tokenize_dataset (truncation at max_length), predict_logits and probabilities_from_logits."""
    from datasets import Dataset, DatasetDict

    split = DatasetDict({"train": Dataset.from_dict({"text": list(texts)})})  # tokenize_dataset expects a "train" split
    tokenized = common.tokenize_dataset(split, tokenizer)["train"]
    return probabilities_from_logits(predict_logits(model, tokenizer, tokenized, work_dir), temperature)


# ---------- Metrics ----------
def split_metrics(split, y_true, probabilities):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    y_pred = probabilities.argmax(axis=1)
    row = {
        "split": split,
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS),
    }
    for label, spec in common.CONFIG["labels"].items():
        row[f"{spec['name']}_error_rate"] = float(np.mean(y_pred[y_true == label] != label))
    row["brier_score"] = multiclass_brier_score(y_true, probabilities)
    row["log_loss"] = log_loss(y_true, probabilities, labels=LABELS)
    return row


def human_vs_ai(y_true, probabilities):
    """Notebook method: drop AI-assisted texts, renormalise P(human) and P(AI), threshold 0.5."""
    import numpy as np
    from sklearn.metrics import confusion_matrix

    mask = y_true != 2
    p = probabilities[mask]
    ai_prob = p[:, 1] / (p[:, 0] + p[:, 1])
    y_pred = (ai_prob >= 0.5).astype(np.int64)
    return confusion_matrix(y_true[mask], y_pred, labels=[1, 0])  # rows/cols: AI, Human


def length_bias(display_name, texts, y_true, probabilities):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score

    y_pred = probabilities.argmax(axis=1)
    words = np.asarray([len(t.split()) for t in texts])
    rows = []
    for lower, upper, label in LENGTH_BINS:
        upper = np.inf if upper is None else upper
        in_bin = (words >= lower) & (words <= upper)
        y_bin, pred_bin, prob_bin = y_true[in_bin], y_pred[in_bin], probabilities[in_bin]
        if len(y_bin) == 0:
            continue
        h, a, s = y_bin == 0, y_bin == 1, y_bin == 2
        rows.append({
            "model": display_name,
            "length_bin": label,
            "samples": len(y_bin),
            "human_samples": int(h.sum()),
            "ai_samples": int(a.sum()),
            "ai_assisted_samples": int(s.sum()),
            "accuracy": accuracy_score(y_bin, pred_bin),
            "mean_human_score": prob_bin[h, 0].mean() * 100,
            "mean_ai_score": prob_bin[a, 1].mean() * 100,
            "mean_ai_assisted_score": prob_bin[s, 2].mean() * 100,
            "human_error_rate": np.mean(pred_bin[h] != 0),
            "ai_error_rate": np.mean(pred_bin[a] != 1),
            "ai_assisted_error_rate": np.mean(pred_bin[s] != 2),
        })
    return pd.DataFrame(rows)


# ---------- Figures (same style as the notebooks, saved as PNG) ----------
def plot_confusion(matrix, tick_labels, path, cell_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(tick_labels)
    fig, ax = plt.subplots(figsize=(7, 7) if n == 3 else (6, 6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for r in range(n):
        for c in range(n):
            value = matrix[r, c]
            color = "white" if value > matrix.max() / 2 else "#222222"
            if cell_labels is None:
                ax.text(c, r, f"{value:,}", ha="center", va="center", color=color, fontsize=22, fontweight="bold")
            else:
                ax.text(c, r - 0.08, f"{value:,}", ha="center", va="center", color=color, fontsize=24)
                ax.text(c, r + 0.14, cell_labels[r][c], ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(n), labels=tick_labels)
    ax.set_yticks(range(n), labels=tick_labels)
    ax.set_xlabel("Model prediction", fontsize=13, fontweight="bold", labelpad=15)
    ax.set_ylabel("Actual label", fontsize=13, fontweight="bold", labelpad=15)
    if cell_labels is not None:
        ax.axhline(0.5, color="#d0d0d0", linewidth=1)
        ax.axvline(0.5, color="#d0d0d0", linewidth=1)
    ax.tick_params(axis="both", labelsize=11, pad=7)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- Main ----------
def evaluate(name: str, from_hub: bool = False) -> dict:
    import tempfile

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    spec = common.model_spec(name)
    cfg = common.CONFIG["training"]

    if from_hub:
        model_ref, config_dir = spec["hf_repo"], common.new_run_dir(name)
    else:
        local = common.artifact_dir(name)
        if not local.exists():
            raise FileNotFoundError(f"No trained model at {local}. Train it first or use --from-hub.")
        model_ref, config_dir = str(local), local

    out = common.new_run_dir(name)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {spec['display_name']} from {model_ref} ...")
    dataset = common.load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForSequenceClassification.from_pretrained(model_ref)
    tokenized = common.tokenize_dataset(dataset, tokenizer)

    with tempfile.TemporaryDirectory() as work_dir:
        logits = {s: predict_logits(model, tokenizer, tokenized[s], work_dir)
                  for s in ("train", "validation", "test")}
    y = {s: np.asarray(dataset[s]["label"], dtype=np.int64) for s in logits}

    # Calibration on the validation split
    temperature = fit_temperature(logits["validation"], y["validation"])
    print(f"Learned temperature: {temperature:.4f}")
    calibration = pd.DataFrame([
        {"model": label,
         "brier_score": multiclass_brier_score(y["validation"], p),
         "log_loss": log_loss(y["validation"], p, labels=LABELS)}
        for label, p in [("Uncalibrated", probabilities_from_logits(logits["validation"])),
                         ("Temperature-scaled", probabilities_from_logits(logits["validation"], temperature))]
    ])
    calibration.to_csv(out / f"{name}-calibration.csv", index=False)

    probs = {s: probabilities_from_logits(logits[s], temperature) for s in logits}
    metrics = pd.DataFrame([split_metrics(s, y[s], probs[s]) for s in ("train", "validation", "test")])
    metrics.to_csv(out / f"{name}-metrics.csv", index=False)
    print(metrics[["split", "accuracy", "macro_f1"]].to_string(index=False))

    # 3-class confusion matrix (test)
    display = [v["display"] for v in common.CONFIG["labels"].values()]
    cm3 = confusion_matrix(y["test"], probs["test"].argmax(axis=1), labels=LABELS)
    pd.DataFrame(cm3, index=[f"true {d}" for d in display], columns=[f"pred {d}" for d in display]) \
        .to_csv(out / f"{name}-3-class-confusion-matrix.csv")
    plot_confusion(cm3, ["Human", "AI", "AI-assisted"], out / f"{name}-confmat-3class.png")

    # 2-class Human vs AI (test, AI-assisted left out)
    cm2 = human_vs_ai(y["test"], probs["test"])
    pd.DataFrame(cm2, index=["true AI", "true Human"], columns=["pred AI", "pred Human"]) \
        .to_csv(out / f"{name}-confmat-human-ai.csv")
    plot_confusion(cm2, ["AI", "Human"], out / f"{name}-confmat-human-ai.png",
                   cell_labels=[["True positive (TP)", "False negative (FN)"],
                                ["False positive (FP)", "True negative (TN)"]])

    # Length bias (test)
    length_bias(spec["display_name"], dataset["test"]["text"], y["test"], probs["test"]) \
        .to_csv(out / f"{name}-length-bias-3class.csv", index=False)

    # detector-config.json, same keys as the notebooks
    training_info = {}
    info_path = common.artifact_dir(name) / "training-info.json"
    if not from_hub and info_path.exists():
        training_info = json.loads(info_path.read_text(encoding="utf-8"))
    detector_config = {
        "dataset": common.CONFIG["dataset_path"],
        "base_model": spec["base_model"],
        "label_mapping": {"human": 0, "ai": 1, "ai_assisted": 2},
        "max_length": cfg["max_length"],
        "calibration": "temperature scaling",
        "temperature": temperature,
        "best_validation_accuracy": float(accuracy_score(y["validation"], logits["validation"].argmax(axis=1))),
        "best_checkpoint": training_info.get("best_checkpoint"),
        "random_state": cfg["seed"],
        "python_version": platform.python_version(),
        "package_versions": {p: version(p) for p in ["datasets", "numpy", "scikit-learn", "torch", "transformers"]},
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "detector-config.json").write_text(json.dumps(detector_config, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {out}")
    return {"metrics": metrics, "temperature": temperature}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help=f"one of: {', '.join(common.MODEL_NAMES)}")
    parser.add_argument("--from-hub", action="store_true", help="load the fine-tuned model from Hugging Face")
    args = parser.parse_args(argv)
    try:
        common.validate_model_name(args.model)
    except common.UnknownModelError as e:
        parser.error(str(e))
    evaluate(args.model, from_hub=args.from_hub)


if __name__ == "__main__":
    main()
