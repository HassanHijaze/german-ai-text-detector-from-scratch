"""Run the whole step 13 model comparison from the saved prediction CSVs.

    python scripts/run.py

Reads the five *-test-predictions.csv files in predictions/ and saves the six
figures to results/. No GPU or models needed.
"""

import analyze
import plot
from common import MODELS, prediction_path


def main():
    missing = [prediction_path(key).name for key in MODELS if not prediction_path(key).exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files in predictions/: {', '.join(missing)}")

    print("== Step 1/2: analyze")
    results = analyze.main()

    print("\n== Step 2/2: plot")
    plot.main(results)
    print("\nDone.")


if __name__ == "__main__":
    main()