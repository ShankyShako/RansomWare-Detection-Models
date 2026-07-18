"""PyTorch deep models with configurable classification heads.

Models (all share one trunk and emit one or more prediction heads):
  * ``dnn``         - multi-layer perceptron
  * ``cnn``         - 1-D convolutional network
  * ``transformer`` - compact RoBERTa-style encoder (distinct weights per layer)
  * ``albert``      - ALBERT-style encoder: **cross-layer parameter sharing** +
                      **factorized embeddings**. The shared layer acts as a
                      strong regulariser, which tends to do well on this small
                      dataset.

The three heads are:
    binary  (2 classes)  group (5 classes)  specific (12 classes)
Any subset can be trained via ``--heads`` (e.g. ``--heads binary,group``) to see
how the level of supervision affects results.

Overfitting controls: internal validation split with **early stopping on
validation macro-F1 + best-weight restore**, AdamW weight decay, dropout, and
label smoothing, so training longer never degrades the saved model.

Requires PyTorch. Examples:
    python src/deep_models.py --model albert
    python src/deep_models.py --model albert --heads binary,group
"""
from __future__ import annotations

import argparse
import copy
import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyTorch is required: pip install torch") from e

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from data import load_raw, split_and_select

# name -> number of classes
HEADS = {"binary": 2, "group": 5, "specific": 12}
ALL_HEADS = tuple(HEADS)


def _make_heads(dim, heads):
    return nn.ModuleDict({k: nn.Linear(dim, HEADS[k]) for k in heads})


class MultiHeadDNN(nn.Module):
    def __init__(self, in_dim, heads=ALL_HEADS, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.heads = _make_heads(128, heads)

    def forward(self, x):
        z = self.trunk(x)
        return {k: h(z) for k, h in self.heads.items()}


class MultiHeadCNN(nn.Module):
    """Treat the feature vector as a 1-D signal and convolve over it."""

    def __init__(self, in_dim, heads=ALL_HEADS, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(32),
        )
        self.flat = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 128), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.heads = _make_heads(128, heads)

    def forward(self, x):
        z = self.flat(self.conv(x.unsqueeze(1)))
        return {k: h(z) for k, h in self.heads.items()}


class _Tokenizer(nn.Module):
    """Chunk a feature vector into tokens + a learned [CLS] and positions."""

    def __init__(self, in_dim, n_tokens, d_model, embed_dim=None):
        super().__init__()
        self.n_tokens = n_tokens
        self.chunk = math.ceil(in_dim / n_tokens)
        self.pad = self.chunk * n_tokens - in_dim
        if embed_dim:  # factorized embedding (ALBERT): chunk -> E -> H
            self.proj = nn.Sequential(nn.Linear(self.chunk, embed_dim),
                                      nn.Linear(embed_dim, d_model))
        else:
            self.proj = nn.Linear(self.chunk, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        b = x.size(0)
        if self.pad:
            x = F.pad(x, (0, self.pad))
        tok = self.proj(x.view(b, self.n_tokens, self.chunk))
        return torch.cat([self.cls.expand(b, -1, -1), tok], dim=1) + self.pos


class TransformerTab(nn.Module):
    """Compact RoBERTa-style encoder: independent weights per layer."""

    def __init__(self, in_dim, heads=ALL_HEADS, n_tokens=16, d_model=128,
                 nhead=4, layers=2, ff=256, dropout=0.3):
        super().__init__()
        self.tok = _Tokenizer(in_dim, n_tokens, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.drop = nn.Dropout(dropout)
        self.heads = _make_heads(d_model, heads)

    def forward(self, x):
        z = self.drop(self.encoder(self.tok(x))[:, 0])  # [CLS]
        return {k: h(z) for k, h in self.heads.items()}


class AlbertTab(nn.Module):
    """ALBERT-style encoder: ONE encoder layer reused across depth (parameter
    sharing) plus a factorized token embedding. The weight sharing keeps the
    parameter count low and strongly regularises training."""

    def __init__(self, in_dim, heads=ALL_HEADS, n_tokens=16, d_model=128,
                 embed_dim=64, nhead=4, layers=4, ff=256, dropout=0.3):
        super().__init__()
        self.tok = _Tokenizer(in_dim, n_tokens, d_model, embed_dim=embed_dim)
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                                batch_first=True)
        self.n_layers = layers
        self.drop = nn.Dropout(dropout)
        self.heads = _make_heads(d_model, heads)

    def forward(self, x):
        h = self.tok(x)
        for _ in range(self.n_layers):   # same weights every layer (ALBERT)
            h = self.layer(h)
        z = self.drop(h[:, 0])           # [CLS]
        return {k: head(z) for k, head in self.heads.items()}


MODELS = {"dnn": MultiHeadDNN, "cnn": MultiHeadCNN,
          "transformer": TransformerTab, "albert": AlbertTab}


def _weights(y, n, device):
    w = compute_class_weight("balanced", classes=np.arange(n), y=y)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _macro_f1(model, X, y_val, heads, device):
    model.eval()
    with torch.no_grad():
        out = model(X)
    return float(np.mean([
        f1_score(y_val[k], out[k].argmax(1).cpu().numpy(), average="macro")
        for k in heads
    ]))


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU (M1/M2/M3/M4)
        return torch.device("mps")
    return torch.device("cpu")


def train(model_name="dnn", heads=ALL_HEADS, epochs=80, batch=64, lr=1e-3,
          weight_decay=1e-2, patience=10, label_smoothing=0.05, k_features=1000,
          verbose=True):
    heads = tuple(heads)
    device = _device()
    X, y_bin, y_grp, y_spec = load_raw()
    (Xtr_all, Xte), ys, meta = split_and_select(X, [y_bin, y_grp, y_spec], k_features)
    y_by_head = {"binary": ys[0], "group": ys[1], "specific": ys[2]}
    if verbose:
        print(f"features {meta['n_features_in']} -> {meta['n_features_out']} | "
              f"device {device} | heads {list(heads)}")

    # Carve a validation set out of the training data (stratified on binary).
    tr, va = train_test_split(np.arange(len(Xtr_all)), test_size=0.2,
                              random_state=42, stratify=ys[0][0])
    Xtr = torch.tensor(Xtr_all[tr], device=device)
    Xva = torch.tensor(Xtr_all[va], device=device)
    ytr = {k: torch.tensor(y_by_head[k][0][tr], device=device) for k in heads}
    yval = {k: y_by_head[k][0][va] for k in heads}

    losses = {k: nn.CrossEntropyLoss(
        weight=_weights(y_by_head[k][0][tr], HEADS[k], device),
        label_smoothing=label_smoothing) for k in heads}

    model = MODELS[model_name](Xtr.shape[1], heads=heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(Xtr, *[ytr[k] for k in heads]),
                        batch_size=batch, shuffle=True)

    best_f1, best_state, wait, best_ep = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        for batch_data in loader:
            xb, labels = batch_data[0], dict(zip(heads, batch_data[1:]))
            opt.zero_grad()
            out = model(xb)
            loss = sum(losses[k](out[k], labels[k]) for k in heads)
            loss.backward()
            opt.step()
        val_f1 = _macro_f1(model, Xva, yval, heads, device)
        if val_f1 > best_f1:
            best_f1, best_state, wait, best_ep = val_f1, copy.deepcopy(model.state_dict()), 0, ep
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best-on-validation weights
    if verbose:
        print(f"  best validation macro-F1 {best_f1:.3f} @ epoch {best_ep}")

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, device=device))
    metrics = {}
    for k in heads:
        pred = out[k].argmax(1).cpu().numpy()
        true = y_by_head[k][1]
        metrics[k] = {
            "accuracy": round(float(accuracy_score(true, pred)), 4),
            "macro_f1": round(float(f1_score(true, pred, average="macro")), 4),
        }
        if verbose:
            print(f"  {k:9s} acc={metrics[k]['accuracy']:.3f}  "
                  f"macroF1={metrics[k]['macro_f1']:.3f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="dnn")
    ap.add_argument("--heads", default="binary,group,specific",
                    help="comma-separated subset of: binary,group,specific")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    chosen = [h.strip() for h in args.heads.split(",") if h.strip()]
    print(f"\n=== {args.model.upper()} | heads={chosen} ===")
    train(args.model, heads=chosen, epochs=args.epochs)
