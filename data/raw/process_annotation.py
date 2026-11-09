import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# 1. Konfigurasi (Pastikan sesuai dengan temuan 7 kolom di atas)
LABEL_COLS = [
    "Price Direction Up",
    "Price Direction Constant",
    "Price Direction Down",
    "Asset Comparision",
    "Past Information",
    "Future Information",
    "Price Sentiment"
]
PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = Path(__file__).resolve().with_name("temporal_test.csv")
ANNOTATION_DIR = PROJECT_DIR / "data" / "annotation"
RAW_DIR = PROJECT_DIR / "data" / "raw"

# 2. Baca data
print("Membaca data...")
try:
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='cp1252')

# 3. Buat file annotator (Simulasi)
# CATATAN: Dalam skenario nyata, bagikan file template kosong ke 3 anotator manusia.
# Di sini, kita gunakan data Anda sebagai dasar (Annotator A), 
# dan buat variasi kecil pada B dan C untuk mendemonstrasikan mekanisme disagreement.
ann_A = df.copy()
ann_B = df.copy()
ann_C = df.copy()

# Simulasi ketidaksepakatan minor untuk pengujian logika konsensus
ann_B.at[2, 'Price Direction Constant'] = 0  # A=1, B=0, C=1 -> Konsensus 1 (Majority)
ann_C.at[3, 'Price Sentiment'] = 'neutral'   # A=positive, B=positive, C=neutral -> Konsensus 'positive'

# Buat direktori jika belum ada
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

ann_A.to_csv(ANNOTATION_DIR / "ann_A.csv", index=False, sep=';')
ann_B.to_csv(ANNOTATION_DIR / "ann_B.csv", index=False, sep=';')
ann_C.to_csv(ANNOTATION_DIR / "ann_C.csv", index=False, sep=';')
print(f"File ann_A.csv, ann_B.csv, dan ann_C.csv berhasil dibuat di {ANNOTATION_DIR}")

# 4 & 5. Simulasi logika kappa.py (Voting Mayoritas)
disagreements = []
consensus_rows = []

for idx, row in df.iterrows():
    val_a = [row[col] for col in LABEL_COLS]
    val_b = [ann_B.iloc[idx][col] for col in LABEL_COLS]
    val_c = [ann_C.iloc[idx][col] for col in LABEL_COLS]
    
    consensus_vals = []
    disagreed_cols = []
    
    for i, col in enumerate(LABEL_COLS):
        v_a, v_b, v_c = val_a[i], val_b[i], val_c[i]
        
        # Cek ketidaksepakatan
        if not (v_a == v_b == v_c):
            disagreed_cols.append(col)
        
        # Logika Voting Mayoritas
        if isinstance(v_a, (int, np.integer)) or (isinstance(v_a, str) and v_a.isdigit()):
            vals = [int(v_a), int(v_b), int(v_c)]
            # "seri dihitung sebagai 1" sesuai instruksi Anda
            consensus = 1 if sum(vals) >= 2 else 0
        else:
            # Untuk kolom kategorikal (Price Sentiment)
            vals = [str(v_a), str(v_b), str(v_c)]
            counts = Counter(vals)
            max_count = max(counts.values())
            modes = [k for k, v in counts.items() if v == max_count]
            consensus = modes[0] # Ambil mode pertama jika seri
            
        consensus_vals.append(consensus)
        
    # Catat disagreement
    if disagreed_cols:
        disagreements.append({
            'Index': idx,
            'Dates': row['Dates'],
            'News_Snippet': str(row['News'])[:60] + '...',
            'Disagreed_Columns': ', '.join(disagreed_cols),
            'Annotator_A': ', '.join([str(val_a[i]) for i, col in enumerate(LABEL_COLS) if col in disagreed_cols]),
            'Annotator_B': ', '.join([str(val_b[i]) for i, col in enumerate(LABEL_COLS) if col in disagreed_cols]),
            'Annotator_C': ', '.join([str(val_c[i]) for i, col in enumerate(LABEL_COLS) if col in disagreed_cols])
        })
        
    # Bangun baris konsensus
    cons_row = {col: row[col] for col in ['Dates', 'URL', 'News']}
    for i, col in enumerate(LABEL_COLS):
        cons_row[col] = consensus_vals[i]
    consensus_rows.append(cons_row)

# 6. Simpan hasil akhir
consensus_df = pd.DataFrame(consensus_rows)
disagreements_df = pd.DataFrame(disagreements)

consensus_df.to_csv(RAW_DIR / "temporal_test.csv", index=False, sep=';')
disagreements_df.to_csv(ANNOTATION_DIR / "disagreements.csv", index=False, sep=';')

print(f"Konsensus berhasil dibuat: {RAW_DIR / 'temporal_test.csv'} ({len(consensus_df)} baris)")
print(f"Ditemukan {len(disagreements_df)} baris ketidaksepakatan di: {ANNOTATION_DIR / 'disagreements.csv'}")

if len(disagreements_df) > 0:
    print("\n🔍 Contoh Ketidaksepakatan yang Diselesaikan:")
    print(disagreements_df[['Index', 'Disagreed_Columns', 'Annotator_A', 'Annotator_B', 'Annotator_C']].to_string(index=False))