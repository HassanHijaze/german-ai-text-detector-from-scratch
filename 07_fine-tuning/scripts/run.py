"""Train and evaluate the fine-tuned detector models.

    python scripts/run.py mmbert-base                   train + evaluate one model
    python scripts/run.py all                           all five models, one after another
    python scripts/run.py mmbert-base --evaluate-only   skip training, evaluate the model from Hugging Face

Run from 07_fine-tuning/. Evaluation results go to results/<name>/new-run/.
Valid model names are listed in config.yaml (models section).
"""

import argparse
import sys

import common


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help=f"'all' or one of: {', '.join(common.MODEL_NAMES)}")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="skip training and evaluate the published model from Hugging Face")
    args = parser.parse_args(argv)
    if args.model != "all":
        try:
            common.validate_model_name(args.model)
        except common.UnknownModelError as e:
            parser.error(str(e))
    return args


def main(argv=None):
    args = parse_args(argv)
    names = common.MODEL_NAMES if args.model == "all" else [args.model]

    import evaluate  # heavy imports only after the arguments are valid
    import train

    for i, name in enumerate(names, 1):
        print(f"\n===== [{i}/{len(names)}] {common.model_spec(name)['display_name']} =====")
        if args.evaluate_only:
            evaluate.evaluate(name, from_hub=True)
        else:
            train.train(name)
            evaluate.evaluate(name, from_hub=False)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
