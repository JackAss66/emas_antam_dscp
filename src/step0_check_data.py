import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Pastikan ini mengarah ke folder data/raw di proyek emas_antam
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES_TO_CHECK = {
    "antam": RAW_DIR / "harga_emas_2015_2019.csv",
    "emas_dunia": RAW_DIR / "harga_emas_harian_net_2015_2019.csv",
    "kurs": RAW_DIR / "kurs_usd_idr_2015_2019.csv",
    "berita": RAW_DIR / "gold_news_sinha_khandait.csv"
}

def check_file(file_key, filepath):
    print(f"\n{'='*60}")
    print(f"MEMERIKSA FILE: {file_key}")
    print(f"Lokasi: {filepath}")
    print(f"{'='*60}")
    
    if not filepath.exists():
        print("⚠️ PERINGATAN: File tidak ditemukan!")
        return
    
    try:
        if file_key == "berita":
            df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        else:
            df = pd.read_csv(filepath, encoding='utf-8')
            
        print(f"✅ Berhasil dibaca. Bentuk: {df.shape}")
        print(f"📋 Nama Kolom: {list(df.columns)}")
        
        # Deteksi kolom tanggal
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'tanggal' in col.lower() or 'tgl' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            print(f"📅 Kolom Tanggal Terdeteksi: '{date_col}'")
            try:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                print(f"📆 Rentang Tanggal: {min_date.date()} s.d. {max_date.date()}")
            except Exception as e:
                print(f"⚠️ Gagal memparse tanggal: {e}")
        else:
            print("⚠️ Tidak ada kolom yang mengandung kata 'date', 'tanggal', atau 'tgl'. Periksa manual.")
            
        print(f"\n👀 3 Baris Pertama:")
        print(df.head(3).to_string())
        
    except Exception as e:
        print(f"❌ ERROR saat membaca file: {e}")

if __name__ == "__main__":
    print("🚀 MEMULAI PEMERIKSAAN DATA MENTAH (PROYEK EMAS_ANTAM)...\n")
    for key, path in FILES_TO_CHECK.items():
        check_file(key, path)
    print("\n✅ PEMERIKSAAN SELESAI. Silakan salin nama kolom dan rentang tanggal di atas untuk melengkapi src/config.py")