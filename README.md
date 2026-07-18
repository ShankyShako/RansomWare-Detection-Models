# Ransomware Detection Models

A comparative study of deep-learning and transformer models for detecting and
classifying ransomware from Windows API-call features. Each model is evaluated
under three feature-engineering setups (a plain baseline, an autoencoder, and
K-Means clustering) and predicts at three levels of granularity.

## Classification levels

Every model produces predictions at three levels simultaneously:

| Level        | Classes | Meaning                                   |
| ------------ | ------- | ----------------------------------------- |
| **Binary**   | 0–1     | Benign (0) vs. malicious (1)              |
| **Group**    | 0–4     | Coarse family groups (goodware + 4 bins)  |
| **Specific** | 0–11    | Individual family (see table below)       |

### Family labels

| ID | Family          | ID | Family          |
| -- | --------------- | -- | --------------- |
| 0  | Goodware        | 6  | Locker          |
| 1  | Critroni        | 7  | MATSNU          |
| 2  | CryptLocker     | 8  | PGPCODER        |
| 3  | CryptoWall      | 9  | Reveton         |
| 4  | KOLLAH          | 10 | TeslaCrypt      |
| 5  | Kovter          | 11 | Trojan-Ransom   |

Group labels bin the families: `1–3 → 1`, `4–6 → 2`, `7–9 → 3`, `10–11 → 4`,
and goodware stays `0`.

## Repository structure

```
.
├── Control/           # Baseline: models trained on raw features
├── AutoEncoder/       # Features passed through an autoencoder first
├── KMeansCluster/     # K-Means cluster assignments appended to features
├── src/               # Clean, reproducible pipeline (see below)
├── results/           # Generated metrics (metrics.json, RESULTS.md)
├── RansomwareData.csv # Dataset (not tracked in git — see below)
├── requirements.txt
└── README.md
```

Each experiment folder contains the same five models, so results are directly
comparable across feature-engineering strategies:

| Model   | File suffix | Description                        |
| ------- | ----------- | ---------------------------------- |
| DNN     | `*DNN`      | Deep (fully-connected) network     |
| CNN     | `*CNN`      | 1-D convolutional network          |
| CNT     | `*CNT`      | Convolutional + transformer hybrid |
| ALBERT  | `*ALBERT`   | ALBERT transformer                 |
| RoBERTa | `*RoBERTa`  | RoBERTa transformer                |

## Dataset

Features are extracted from the
[RISS ransomware dataset (2016)](https://github.com/rissgrouphub/ransomwaredataset2016).
`RansomwareData.csv` has one row per sample: column 0 is a sample ID, column 1
is the binary label, column 2 is the specific-family label, and the remaining
columns are binary API-call / behavioral features.

The CSV is ~91 MB and is **not tracked in git** (see `.gitignore`). Download it
from the source above, or place your own copy in the repository root. To point
the notebooks at a different location, set the `DATA_PATH` environment variable.

## Setup

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Place RansomwareData.csv in the repo root (or set DATA_PATH)
export DATA_PATH=/path/to/RansomwareData.csv   # optional

# 3. Launch Jupyter and open any notebook
jupyter notebook
```

The notebooks were originally developed on Google Colab. The Colab
`drive.mount(...)` calls are left in place but commented out, so the notebooks
run locally by default and can be re-enabled for Colab in one line.

## Reproducible pipeline (`src/`)

Alongside the exploratory notebooks, `src/` holds a clean, runnable pipeline:

- `data.py` — loading, three-level label construction, stratified split, and
  chi²-based feature selection (30,967 → 1,000 features) fit on the training
  split only to avoid leakage.
- `run_experiments.py` — trains a Random Forest, a class-balanced Logistic
  Regression, and an MLP (feed-forward neural net) at all three levels, then
  writes `results/RESULTS.md` and `results/metrics.json`. Runs on CPU in
  seconds.
- `deep_models.py` — PyTorch multi-head **DNN**, **1-D CNN**, a compact
  **RoBERTa-style transformer**, and an **ALBERT-style transformer**
  (cross-layer parameter sharing + factorized embeddings). All share one trunk,
  emit one or more heads, and use class-weighted loss, an internal validation
  split with **early stopping on validation macro-F1 + best-weight restore**,
  AdamW weight decay, dropout, and label smoothing — so training longer never
  degrades the saved model. Which heads to train is configurable
  (`--heads binary,group`). Requires `torch`; run locally or on Colab.
- `head_ablation.py` — trains one model under `binary`, then `binary+group`,
  then all three heads, and writes `results/HEAD_ABLATION.md` so you can see
  whether adding heads helps or hurts each level.

> **Why ALBERT tends to win here.** ALBERT reuses a single encoder layer across
> depth, so its parameter count stays tiny regardless of depth. On a ~1.5k-sample
> dataset that weight sharing is a strong regulariser, which is why it resists
> the overfitting that a full-size RoBERTa suffers.

  > **Note on the transformer / overfitting.** The transformer is intentionally
  > small (2 layers, 4 heads, hidden 128). A full RoBERTa-size model (12 layers,
  > ~100M parameters) trained from scratch memorises this ~1.5k-sample dataset
  > within a few epochs, which is why accuracy *falls* the longer you train it.
  > Matching model capacity to the data — plus early stopping and weight decay —
  > is what fixes that.

```bash
DATA_PATH=/path/to/RansomwareData.csv python src/run_experiments.py
DATA_PATH=/path/to/RansomwareData.csv python src/deep_models.py --model cnn
```

### Quick start on a Mac

With `RansomwareData.csv` in the repo root, `run.sh` handles the virtual
environment and dependencies for you. From a Terminal in the project folder:

```bash
./run.sh setup      # one-time: creates .venv and installs everything (incl. PyTorch)
./run.sh all        # runs EVERY model and writes the report  <-- the main command
```

That's it. `run.sh all` trains the scikit-learn baselines **and** the PyTorch
DNN, CNN, RoBERTa-style transformer, and ALBERT, then writes a single
consolidated report to:

- `results/RESULTS.md` — human-readable summary tables
- `results/metrics.json` — machine-readable, timestamped

Because `results/` lives inside the project folder, the output is easy to share
or review after each run. You can also run pieces individually:

```bash
./run.sh baseline     # scikit-learn models only (fast)
./run.sh dnn          # PyTorch deep neural network only
./run.sh cnn          # PyTorch 1-D convolutional network only
./run.sh transformer  # RoBERTa-style transformer only
./run.sh albert       # ALBERT-style transformer only
./run.sh ablation     # head-combination study -> results/HEAD_ABLATION.md
```

Any single deep model accepts `--heads` to choose which levels to train, e.g.
`./run.sh albert --heads binary,group`.

The PyTorch models automatically use the Apple Silicon GPU (M1/M2/M3/M4) via
Metal/MPS — no extra configuration needed.

### Latest results

Test-set scores from the latest run (`./run.sh all --heads binary,group`, run
on an Apple Silicon GPU). Feature selection both speeds things up and lifts
accuracy — binary went from ~94% on the raw 30k features to ~98% on the top
1,000 selected features. Models are sorted by macro-F1 within each level.

**Binary** (benign vs. malicious)

| Model | Accuracy | Macro-F1 |
| ----- | -------- | -------- |
| Logistic Regression | 0.984 | 0.983 |
| DNN (PyTorch) | 0.984 | 0.983 |
| Transformer / RoBERTa (PyTorch) | 0.980 | 0.979 |
| Random Forest | 0.977 | 0.976 |
| MLP | 0.974 | 0.972 |
| ALBERT (PyTorch) | 0.964 | 0.961 |
| CNN (PyTorch) | 0.938 | 0.935 |

**Group** (0–4)

| Model | Accuracy | Macro-F1 |
| ----- | -------- | -------- |
| MLP | 0.911 | 0.731 |
| Random Forest | 0.898 | 0.712 |
| Logistic Regression | 0.889 | 0.686 |
| DNN (PyTorch) | 0.885 | 0.682 |
| ALBERT (PyTorch) | 0.862 | 0.646 |
| Transformer / RoBERTa (PyTorch) | 0.853 | 0.622 |
| CNN (PyTorch) | 0.816 | 0.611 |

At the binary level the top models are effectively tied (~0.98) — the spread is
within run-to-run noise, since the PyTorch models are not seed-fixed, so the
exact ranking among the close models will shift slightly between runs. On the
harder group level the simpler models (MLP, Random Forest) currently edge out
the transformers.

The **specific** (12-family) level was not part of this run. Run `./run.sh all`
(no `--heads`) to add it. In earlier full runs the macro-F1 there was much
lower (~0.55), driven by class scarcity rather than model capacity: several
families have only a handful of samples (e.g. PGPCODER and TeslaCrypt have 1–2
in the test split), so they are effectively unlearnable without more data,
oversampling (e.g. SMOTE), or collapsing rare families. See
`results/RESULTS.md` for the full per-class breakdown once generated.

## License

Released under the [MIT License](LICENSE).
