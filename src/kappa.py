"""FASE 1.3 - Cohen's Kappa antar anotator + pembuatan label konsensus.

Format tiap file anotator (CSV): kolom Dates, News, dan 9 kolom label (0/1),
baris HARUS sama & urutannya sama untuk semua anotator.

  python src/kappa.py data/annotation/ann_A.csv data/annotation/ann_B.csv data/annotation/ann_C.csv

Output:
  - Kappa per label (semua pasangan anotator) + rata-rata
  - data/raw/temporal_test.csv          (label konsensus, voting mayoritas)
  - data/annotation/disagreements.csv   (baris yang tidak bulat -> diskusikan)
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import DATA_ANN, DATE_COL, LABEL_COLS, RAW_TEMPORAL_CSV, TEXT_COL
from utils import normalize_label_columns, read_csv_file


def main(paths):
    if len(paths) < 2:
        raise SystemExit("Minimal 2 file anotator.")
    anns = [
        normalize_label_columns(read_csv_file(p).dropna(how="all"))
        for p in paths
    ]
    n = len(anns[0])
    for p, a in zip(paths, anns):
        if len(a) != n or not (a[TEXT_COL].values == anns[0][TEXT_COL].values).all():
            raise SystemExit(f"{p}: baris/urutan berbeda dari file pertama.")

    # Kappa per pasangan
    all_k = []
    for (i, a), (j, b) in itertools.combinations(enumerate(anns), 2):
        ks = {}
        for lab in LABEL_COLS:
            x, y = a[lab].values, b[lab].values
            ks[lab] = 1.0 if (x == y).all() else cohen_kappa_score(x, y)  # kappa tak terdefinisi jika konstan
        all_k.append(ks)
        print(f"\nKappa anotator {i + 1} vs {j + 1}:")
        for lab, k in ks.items():
            print(f"  {lab:20s} {k:.3f}")
        print(f"  {'RATA-RATA':20s} {np.mean(list(ks.values())):.3f}")
    overall = np.mean([np.mean(list(k.values())) for k in all_k])
    print(f"\nKappa rata-rata keseluruhan: {overall:.3f}  "
          f"({'>= 0.85 OK' if overall >= 0.85 else '< 0.85 -> perbaiki pedoman anotasi & diskusi ulang'})")

    # Konsensus voting mayoritas (seri -> 1, ditandai untuk ditinjau manual)
    stack = np.stack([a[LABEL_COLS].values for a in anns])  # (n_ann, n, labels)
    mean = stack.mean(0)
    consensus = (mean >= 0.5).astype(int)
    unanimous = ((mean == 0) | (mean == 1)).all(1)

    out = anns[0][[DATE_COL, TEXT_COL]].copy()
    out[LABEL_COLS] = consensus
    out.to_csv(RAW_TEMPORAL_CSV, index=False)
    print(f"\nKonsensus disimpan: {RAW_TEMPORAL_CSV}")

    dis = anns[0].loc[~unanimous, [DATE_COL, TEXT_COL]].copy()
    for k, a in enumerate(anns):
        for lab in LABEL_COLS:
            dis[f"A{k + 1}_{lab}"] = a.loc[~unanimous, lab].values
    dis.to_csv(DATA_ANN / "disagreements.csv", index=False)
    print(f"{int((~unanimous).sum())}/{n} baris tidak bulat -> {DATA_ANN / 'disagreements.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
