import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.evaluate_nlp import evaluate


if __name__ == "__main__":
    evaluate()
