"""Train and evaluate models at all three classification levels.

Runs entirely on CPU with scikit-learn (no GPU or heavy deep-learning
frameworks required). Trains a Random Forest, a class-balanced Logistic
Regression, and a multi-layer perceptron (a feed-forward deep neural network)
for each of the binary / group / specific label sets, then writes a metrics
report to ``results/``.

Usage:
    DATA_PATH=/path/to/RansomwareData.csv python src/run_experiments.py
    python src/run_experiments.py --heads binary,group   # subset of levels
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier

from data import load_raw, split_and_select, FAMILY_NAMES

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEVELS = ["binary", "group", "specific"]


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1
        ),
        "MLP (deep net)": MLPClassifier(
            hidden_layer_sizes=(256, 128), early_stopping=True,
            max_iter=200, random_state=42
        ),
    }


def run_sklearn(heads=LEVELS):
    """Train the scikit-learn models on the selected levels; return results."""
    heads = [h for h in LEVELS if h in heads]  # keep canonical order
    X, y_bin, y_grp, y_spec = load_raw()
    y_by_level = dict(zip(LEVELS, [y_bin, y_grp, y_spec]))

    (X_train, X_test), y_splits, meta = split_and_select(
        X, [y_by_level[h] for h in heads], k_features=1000)
    print(f"Samples: {len(X)} | features {meta['n_features_in']} -> "
          f"{meta['n_features_out']} (chi2-selected)")
    print(f"Train {len(X_train)} / Test {len(X_test)} | levels {heads}\n")

    results = {"meta": meta, "levels": {L: {} for L in heads}}
    for level, (y_tr, y_te) in zip(heads, y_splits):
        print(f"=== {level.upper()} ({len(np.unique(y_tr))} classes) ===")
        for name, model in build_models().items():
            t = time.time()
            model.fit(X_train, y_tr)
            pred = model.predict(X_test)
            results["levels"][level][name] = {
                "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
                "seconds": round(time.time() - t, 1),
            }
            r = results["levels"][level][name]
            print(f"  {name:20s} acc={r['accuracy']:.3f}  "
                  f"macroF1={r['macro_f1']:.3f}  ({r['seconds']}s)")
        print()

    # Detailed per-class report for the specific level, if it was trained.
    if "specific" in heads:
        i = heads.index("specific")
        best = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
        ).fit(X_train, y_splits[i][0])
        results["specific_per_class_report"] = classification_report(
            y_splits[i][1], best.predict(X_test),
            labels=list(range(len(FAMILY_NAMES))),
            target_names=FAMILY_NAMES, zero_division=0,
        )
    return results


def main(heads=LEVELS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_sklearn(heads)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results)
    print("Wrote results/metrics.json and results/RESULTS.md")


def _write_markdown(results):
    lines = ["# Results\n"]
    if results.get("generated_at"):
        lines.append(f"_Generated {results['generated_at']}_\n")
    lines.append(f"Dataset: {results['meta']['n_features_in']} features reduced to "
                 f"{results['meta']['n_features_out']} via chi2 selection.\n")
    for level in LEVELS:
        if not results["levels"].get(level):
            continue  # level not trained in this run
        lines.append(f"\n## {level.capitalize()}\n")
        lines.append("| Model | Accuracy | Macro-F1 |")
        lines.append("| ----- | -------- | -------- |")
        for name, m in results["levels"][level].items():
            lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |")
    if results.get("specific_per_class_report"):
        lines.append("\n## Per-class report (specific families, Random Forest)\n")
        lines.append("```\n" + results["specific_per_class_report"] + "\n```")
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_heads(value):
    chosen = [h.strip() for h in value.split(",") if h.strip()]
    bad = [h for h in chosen if h not in LEVELS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown head(s) {bad}; choose from {LEVELS}")
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=parse_heads, default=LEVELS,
                    help="comma-separated subset of: binary,group,specific")
    args = ap.parse_args()
    main(args.heads)
