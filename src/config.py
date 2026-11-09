from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "outputs"
PRED_DIR, FIG_DIR = OUT / "preds", OUT / "figures"
for p in (RAW, PROC, PRED_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- file mentah
ANTAM_CSV = RAW / "harga_emas_2015_2019.csv"       # Sebenarnya berisi GC=F, akan kita olah
EMAS_DUNIA_CSV = RAW / "harga_emas_harian_net_2015_2019.csv"
KURS_CSV = RAW / "kurs_usd_idr_2015_2019.csv"
BERITA_CSV = RAW / "gold_news_sinha_khandait.csv"

# --- nama kolom mentah (sesuai hasil step0)
# Catatan: File yfinance memiliki baris metadata, jadi kolom tanggalnya terbaca sebagai 'Price'
ANTAM_DATE_COL = "Price"  
ANTAM_PRICE_COL = "Close" 

EMAS_DUNIA_DATE_COL = "Tanggal"
# Kita akan gunakan 'Kurs Tengah BI' untuk satuan 'Gram(gr)' sebagai proksi harga Antam/gram
EMAS_DUNIA_PRICE_COL = "Kurs Tengah BI" 

KURS_DATE_COL = "Price"
KURS_PRICE_COL = "Kurs_yfinance"

# --- nama kolom berita
BERITA_DATE_COL = "Dates"
BERITA_TEXT_COL = "News"
LABEL_NAIK = "Price Direction Up"
LABEL_TURUN = "Price Direction Down"

# --- periode penelitian (IRISAN TANGGAL)
START_DATE = "2015-01-02"
END_DATE = "2019-02-01"

# --- aturan data
MAX_FFILL_DAYS = 3        # isi data kosong ke depan paling lama 3 hari
JUMP_FLAG = 0.05          # tandai perubahan harian di atas 5%

# --- walk-forward dan model
N_BLOCKS = 6
VAL_FRAC = 0.15
WINDOWS = [10, 20, 30]
UNITS = [32, 64]
DROPOUTS = [0.1, 0.3]
SEEDS = [0, 1, 2, 3, 4]
VOL_WINDOW, VOL_PERCENTILE = 20, 75

# --- Definisi Set Fitur (TAMBAHKAN DI PALING BAWAH)
FEATURES_A = ['lag_return_antam', 'lag_return_emas_usd', 'lag_return_kurs']
FEATURES_B = FEATURES_A + ['skor_bersih']