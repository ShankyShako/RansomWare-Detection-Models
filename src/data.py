"""Data loading, label construction, and feature selection.

The dataset (``RansomwareData.csv``) has no header row. Layout per row:
    col 0        -> sample ID (dropped)
    col 1        -> binary label   (0 benign, 1 malicious)
    col 2        -> family label   (0 goodware, 1-11 ransomware families)
    col 3..end   -> binary API-call / behavioral features
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.model_selection import train_test_split

DATA_PATH = os.environ.get("DATA_PATH", "RansomwareData.csv")

FAMILY_NAMES = [
    "Goodware", "Critroni", "CryptLocker", "CryptoWall", "KOLLAH", "Kovter",
    "Locker", "MATSNU", "PGPCODER", "Reveton", "TeslaCrypt", "Trojan-Ransom",
]


def family_to_group(label: int) -> int:
    """Bin the 12 families into 5 coarse groups (0 = goodware)."""
    if label == 0:
        return 0
    if label <= 3:
        return 1
    if label <= 6:
        return 2
    if label <= 9:
        return 3
    return 4


def load_raw(path: str = DATA_PATH):
    """Return (X, y_binary, y_group, y_specific) from the CSV."""
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 3:].to_numpy(dtype=np.float32)
    y_binary = df.iloc[:, 1].to_numpy(dtype=np.int64)
    y_specific = df.iloc[:, 2].to_numpy(dtype=np.int64)
    y_group = np.array([family_to_group(v) for v in y_specific], dtype=np.int64)
    return X, y_binary, y_group, y_specific


def split_and_select(X, ys, k_features=1000, test_size=0.2, seed=42):
    """Stratified split (on the binary label) + chi2 feature selection.

    Feature selection is fit on the training set only to avoid leakage.
    Returns ((X_train, X_test), selector, list_of_(y_train, y_test)).
    """
    idx = np.arange(len(X))
    y_stratify = ys[0]
    tr, te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y_stratify
    )
    X_train, X_test = X[tr], X[te]

    # Drop always-constant columns, then keep the top-k by chi2 vs. binary label.
    vt = VarianceThreshold()
    X_train_v = vt.fit_transform(X_train)
    X_test_v = vt.transform(X_test)

    k = min(k_features, X_train_v.shape[1])
    skb = SelectKBest(chi2, k=k)
    X_train_s = skb.fit_transform(X_train_v, ys[0][tr])
    X_test_s = skb.transform(X_test_v)

    y_splits = [(y[tr], y[te]) for y in ys]
    meta = {"n_features_in": X.shape[1], "n_features_out": k}
    return (X_train_s.astype(np.float32), X_test_s.astype(np.float32)), y_splits, meta
