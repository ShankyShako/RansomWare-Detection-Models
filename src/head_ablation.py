"""Head-combination ablation study.

Trains one model repeatedly under different subsets of the three classification
heads and reports how each level's test metrics change with the amount of
multi-task supervision:

    binary                     (single task)
    binary, group              (two tasks)
    binary, group, specific    (all three)

This answers "does adding the group/specific heads change the binary result?"
Because the heads share a trunk, extra heads can help (useful auxiliary signal)
or hurt (competing objectives) — this measures which.

Usage:
    python src/head_ablation.py --model albert
    ./run.sh ablation                       # defaults to albert

Writes results/HEAD_ABLATION.md and results/head_ablation.json.
"""
from __future__ import annotations

import argparse
import json
import os

from deep_models import train

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
COMBOS = [
    ("binary",),
    ("binary", "group"),
    ("binary", "group", "specific"),
]


def main(model="albert", epochs=80):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for heads in COMBOS:
        print(f"\n### {model.upper()} | heads = {list(heads)} ###")
        metrics = train(model_name=model, heads=heads, epochs=epochs)
        rows.append({"heads": list(heads), "metrics": metrics})

    payload = {"model": model, "combinations": rows}
    with open(os.path.join(RESULTS_DIR, "head_ablation.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = [f"# Head-combination ablation — {model.upper()}\n",
             "Test-set metrics as heads are added to the shared-trunk model. "
             "A metric only appears once its head is trained.\n",
             "| Heads trained | Binary acc | Binary F1 | Group acc | Group F1 | "
             "Specific acc | Specific F1 |",
             "| ------------- | ---------- | --------- | --------- | -------- | "
             "------------ | ----------- |"]
    for r in rows:
        m = r["metrics"]
        def cell(h, key):
            return f"{m[h][key]:.3f}" if h in m else "—"
        lines.append(
            f"| {', '.join(r['heads'])} | {cell('binary','accuracy')} | "
            f"{cell('binary','macro_f1')} | {cell('group','accuracy')} | "
            f"{cell('group','macro_f1')} | {cell('specific','accuracy')} | "
            f"{cell('specific','macro_f1')} |")
    with open(os.path.join(RESULTS_DIR, "HEAD_ABLATION.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {os.path.normpath(RESULTS_DIR)}/HEAD_ABLATION.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dnn", "cnn", "transformer", "albert"],
                    default="albert")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    main(args.model, args.epochs)
