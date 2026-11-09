"""Fungsi bantu yang dipakai banyak skrip."""
import csv
import importlib
import random
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for training only
    torch = None

from config import (BINARY_LABEL_COLS, DATA_PROC, LABEL_COLS,
                    PRICE_SENTIMENT_COL, PRICE_SENTIMENT_LABELS, SEED,
                    STOPWORDS_REMOVE)


def read_csv_file(path):
    """Read CSVs with detected delimiters and common Windows encodings."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=encoding, newline="") as csv_file:
                delimiter = csv.Sniffer().sniff(
                    csv_file.readline(), delimiters=",;\t"
                ).delimiter
            return pd.read_csv(path, sep=delimiter, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"CSV encoding not supported: {path}")


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand categorical price sentiment and validate the nine binary labels."""
    df = df.copy()
    missing = [col for col in BINARY_LABEL_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom label biner tidak ditemukan: {missing}")

    if PRICE_SENTIMENT_COL in df.columns:
        sentiment = (
            df[PRICE_SENTIMENT_COL]
            .astype("string")
            .str.strip()
            .str.lower()
            .fillna("none")
        )
        valid_values = set(PRICE_SENTIMENT_LABELS.values()) | {"none"}
        invalid = sorted(set(sentiment.unique()) - valid_values)
        if invalid:
            raise ValueError(
                f"Kategori {PRICE_SENTIMENT_COL} tidak dikenal: {invalid}; "
                f"gunakan {sorted(valid_values)}"
            )
        for label, category in PRICE_SENTIMENT_LABELS.items():
            df[label] = sentiment.eq(category).astype(int)
    else:
        missing = [col for col in PRICE_SENTIMENT_LABELS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom sentimen sumber atau label hasil one-hot tidak ditemukan: {missing}"
            )

    for col in LABEL_COLS:
        numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if not numeric.isin([0, 1]).all():
            values = numeric[~numeric.isin([0, 1])].unique()[:10]
            raise ValueError(f"Kolom '{col}' harus bernilai 0/1; ditemukan {values}")
        df[col] = numeric.astype(int)
    return df


def set_seed(seed: int = SEED):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(s: str) -> str:
    """Preprocessing untuk Bi-GRU: lowercase -> angka jadi 'num' ->
    hapus tanda baca -> hapus stop-words tertentu (kata arah dipertahankan)."""
    s = str(s).lower()
    s = re.sub(r"\d+([.,]\d+)*", " num ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [t for t in s.split() if t not in STOPWORDS_REMOVE]
    return " ".join(tokens)


def load_processed(name: str):
    """Baca data/processed/<name>.csv; None jika belum ada."""
    p = DATA_PROC / f"{name}.csv"
    return read_csv_file(p) if p.exists() else None


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Metrik multi-label. Macro dihitung hanya pada kelas yang punya
    minimal 1 sampel positif (support > 0) agar tidak bias ke 0."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    m = sup > 0
    return {
        "macro_precision": float(p[m].mean()),
        "macro_recall": float(r[m].mean()),
        "macro_f1": float(f[m].mean()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class_f1": f.tolist(),
        "per_class_support": sup.tolist(),
    }


def pretty(tag: str) -> str:
    if tag == "teacher":
        return "FinBERT (Teacher)"
    if tag == "baseline":
        return "Bi-GRU (No Dist.)"
    return "Bi-GRU (Distill.)"
