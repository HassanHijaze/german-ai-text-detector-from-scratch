"""Fine-tune one model with the shared settings from config.yaml.

    python scripts/train.py mmbert-base      (run from 07_fine-tuning/)

Follows the original notebooks (archive/): same TrainingArguments, early stopping on
validation accuracy, and the best checkpoint is restored at the end. The model and
tokenizer are saved to 07_fine-tuning/artifacts/<name>-3class/.
"""

import argparse
import json

import common


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score

    logits, labels = eval_pred
    return {"accuracy": accuracy_score(labels, np.argmax(logits, axis=1))}


def train(name: str) -> dict:
    import transformers
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    spec = common.model_spec(name)
    cfg = common.CONFIG["training"]
    seed = cfg["seed"]

    dataset = common.load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(spec["base_model"])
    tokenized = common.tokenize_dataset(dataset, tokenizer)

    # Seed before the classification head is created, so the run is repeatable.
    # (The notebooks only seeded inside the Trainer, after the head was created.)
    transformers.set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["base_model"],
        num_labels=len(common.CONFIG["labels"]),
        id2label=common.id2label(),
        label2id=common.label2id(),
    )

    args = TrainingArguments(
        output_dir=str(common.checkpoint_dir(name)),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        save_total_limit=cfg["save_total_limit"],
        report_to="none",
        seed=seed,
        data_seed=seed,
        **common.precision_flags(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=common.data_collator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    print(f"Training {spec['display_name']} ({spec['base_model']}) ...")
    trainer.train()

    out_dir = common.artifact_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Kept for evaluate.py, which writes detector-config.json
    info = {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "epochs_trained": trainer.state.epoch,
    }
    (out_dir / "training-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model and tokenizer to {out_dir}")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help=f"one of: {', '.join(common.MODEL_NAMES)}")
    args = parser.parse_args(argv)
    try:
        common.validate_model_name(args.model)
    except common.UnknownModelError as e:
        parser.error(str(e))
    train(args.model)


if __name__ == "__main__":
    main()
