#!/usr/bin/env bash
# Simple runner for the ransomware-detection models on macOS / Linux.
#
#   ./run.sh setup      # one-time: create venv + install everything (incl. PyTorch)
#   ./run.sh all        # run EVERY model and write results/RESULTS.md  <-- main one
#   ./run.sh baseline   # scikit-learn models only (RF / LogReg / MLP)
#   ./run.sh dnn         # PyTorch multi-head deep neural network only
#   ./run.sh cnn         # PyTorch 1-D convolutional network only
#   ./run.sh transformer # compact RoBERTa-style transformer only
#   ./run.sh albert      # ALBERT-style transformer only
#   ./run.sh ablation    # head-combination study (binary vs +group vs +specific)
#
# Every model command accepts --heads to pick which levels to train, e.g.:
#   ./run.sh all --heads binary,group
#   ./run.sh baseline --heads binary
#   ./run.sh albert --heads binary,group
#
# The PyTorch models automatically use your Mac's GPU (Apple Silicon / MPS).
set -e

cd "$(dirname "$0")"
VENV=.venv
PY="$VENV/bin/python"

# Point at the dataset (override by exporting DATA_PATH before running).
export DATA_PATH="${DATA_PATH:-$(pwd)/RansomwareData.csv}"

case "$1" in
  setup)
    echo "Creating virtual environment in $VENV ..."
    python3 -m venv "$VENV"
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt torch
    echo "Done. Now run:  ./run.sh baseline"
    ;;
  all)
    "$PY" src/run_all.py "${@:2}"
    ;;
  baseline)
    "$PY" src/run_experiments.py "${@:2}"
    ;;
  dnn)
    "$PY" src/deep_models.py --model dnn "${@:2}"
    ;;
  cnn)
    "$PY" src/deep_models.py --model cnn "${@:2}"
    ;;
  transformer|roberta)
    "$PY" src/deep_models.py --model transformer "${@:2}"
    ;;
  albert)
    "$PY" src/deep_models.py --model albert "${@:2}"
    ;;
  ablation)
    "$PY" src/head_ablation.py "${@:2}"
    ;;
  *)
    echo "Usage: ./run.sh {setup|all|baseline|dnn|cnn|transformer|albert|ablation}"
    echo "First time?  Run:  ./run.sh setup   then   ./run.sh all"
    exit 1
    ;;
esac
