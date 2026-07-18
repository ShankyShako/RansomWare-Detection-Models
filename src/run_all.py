"""Run every model and write a single consolidated report to ``results/``.

Runs the scikit-learn baselines always, and the PyTorch DNN / CNN / transformer
/ ALBERT if PyTorch is installed (skipped gracefully otherwise). Produces:
    results/metrics.json   - machine-readable, all models
    results/RESULTS.md     - human-readable summary table

By default all three classification levels are trained; use ``--heads`` to
restrict every model to a subset, e.g. ``--heads binary,group``.

Usage:  python src/run_all.py [--heads binary,group]
        (or  ./run.sh all [--heads binary,group]  from the project root)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

from run_experiments import run_sklearn, _write_markdown, RESULTS_DIR, LEVELS, parse_heads


def _try_deep(heads):
    """Return {label: metrics, ...} for the deep models, or {} if no torch."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\nPyTorch not installed - skipping the deep models. "
              "Run './run.sh setup' to enable them.")
        return {}
    from deep_models import train
    out = {}
    for label, key in [("DNN (PyTorch)", "dnn"), ("CNN (PyTorch)", "cnn"),
                       ("Transformer/RoBERTa (PyTorch)", "transformer"),
                       ("ALBERT (PyTorch)", "albert")]:
        print(f"\n### Training {label} | heads {heads} ###")
        out[label] = train(model_name=key, heads=heads)
    return out


def main(heads=LEVELS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_sklearn(heads)

    for label, m in _try_deep(heads).items():
        for level in LEVELS:
            if level in m:
                results["levels"].setdefault(level, {})[label] = m[level]

    results["generated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results)
    print(f"\nAll done. See {os.path.normpath(RESULTS_DIR)}/RESULTS.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=parse_heads, default=LEVELS,
                    help="comma-separated subset of: binary,group,specific")
    args = ap.parse_args()
    main(args.heads)
