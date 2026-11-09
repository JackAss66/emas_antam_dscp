import pandas as pd
import numpy as np
from pathlib import Path
import sys
import config  # Pastikan config.py berada di folder src yang sama

def clean_yfinance(df, date_col_name, price_col_name):
    """Membersihkan format CSV yfinance yang memiliki baris metadata di awal."""
    # PERBAIKAN: Skip 2 baris pertama (baris 0 = 'Ticker', baris 1 = 'Date')
    df_clean = df.iloc[2:].copy()
    
    # Beri nama kolom yang benar
    df_clean.columns = ['Date', price_col_name] + list(df.columns[2:])
    
    # Konversi ke datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.date
    
    # Ambil hanya kolom yang dibutuhkan
    df_clean = df_clean[['Date', price_col_name]]
    
    # Pastikan kolom harga bertipe numerik
    df_clean[price_col_name] = pd.to_numeric(df_clean[price_col_name], errors='coerce')
    
    return df_clean

def main():
    print("🚀 MEMULAI PEMBERSIHAN DAN PENGGABUNGAN DATA HARIAN...\n")
    
    # 1. Baca Data Antam (GC=F)
    print("📥 Membaca data Antam (GC=F)...")
    df_antam = pd.read_csv(config.ANTAM_CSV)
    df_antam = clean_yfinance(df_antam, config.ANTAM_DATE_COL, config.ANTAM_PRICE_COL)
    df_antam.rename(columns={'Date': 'tanggal', config.ANTAM_PRICE_COL: 'emas_usd_close'}, inplace=True)
    
    # 2. Baca Data Kurs
    print("📥 Membaca data Kurs USD/IDR...")
    df_kurs = pd.read_csv(config.KURS_CSV)
    df_kurs = clean_yfinance(df_kurs, config.KURS_DATE_COL, config.KURS_PRICE_COL)
    df_kurs.rename(columns={'Date': 'tanggal', config.KURS_PRICE_COL: 'kurs_idr'}, inplace=True)
    
    # 3. Baca Data Emas Dunia (Net)
    print("📥 Membaca data Emas Dunia (Net)...")
    df_emas_net = pd.read_csv(config.EMAS_DUNIA_CSV)
    df_emas_net['Tanggal'] = pd.to_datetime(df_emas_net['Tanggal']).dt.date
    
    # Filter hanya satuan Gram untuk mendapatkan proksi harga per gram dalam Rupiah
    df_emas_gram = df_emas_net[df_emas_net['Satuan'] == 'Gram(gr)'].copy()
    
    # Bersihkan string "475.307( -5.964 )" menjadi float 475307.0
    df_emas_gram['Kurs Tengah BI'] = df_emas_gram['Kurs Tengah BI'].astype(str).str.extract(r'([\d\.]+)')[0].str.replace('.', '', regex=False).astype(float)
    
    df_emas_gram.rename(columns={'Tanggal': 'tanggal', 'Kurs Tengah BI': 'proksi_antam_idr'}, inplace=True)
    df_emas_gram = df_emas_gram[['tanggal', 'proksi_antam_idr']]
    
    # 4. Merge Semua Data
    print("🔗 Menggabungkan data berdasarkan tanggal...")
    # Gunakan tanggal dari emas_usd sebagai basis kalender utama (hari perdagangan)
    df_merged = df_antam.merge(df_kurs, on='tanggal', how='left')
    df_merged = df_merged.merge(df_emas_gram, on='tanggal', how='left')
    
    # 5. Filter Periode Penelitian (Irisan)
    print(f"📅 Memfilter data untuk periode: {config.START_DATE} s.d. {config.END_DATE}...")
    
    # PERBAIKAN: Ubah string dari config menjadi datetime.date agar bisa dibandingkan dengan kolom 'tanggal'
    start_date = pd.to_datetime(config.START_DATE).date()
    end_date = pd.to_datetime(config.END_DATE).date()
    
    df_merged = df_merged[(df_merged['tanggal'] >= start_date) & (df_merged['tanggal'] <= end_date)]
    df_merged = df_merged.sort_values('tanggal').reset_index(drop=True)
    
    # 6. Aturan Forward Fill (Maksimal MAX_FFILL_DAYS)
    print(f"🧹 Membersihkan data kosong (Forward Fill maksimal {config.MAX_FFILL_DAYS} hari)...")
    cols_to_ffill = ['kurs_idr', 'proksi_antam_idr']
    for col in cols_to_ffill:
        # Tandai baris yang di-ffill
        df_merged[f'ffill_{col}'] = df_merged[col].isna().astype(int)
        df_merged[col] = df_merged[col].ffill(limit=config.MAX_FFILL_DAYS)
    
    # Buang baris yang masih NaN setelah ffill (misal di awal periode)
    initial_na_count = df_merged.isna().any(axis=1).sum()
    df_merged = df_merged.dropna(subset=['emas_usd_close', 'kurs_idr', 'proksi_antam_idr']).reset_index(drop=True)
    print(f"   ⚠️ {initial_na_count} baris dibuang karena masih mengandung NaN setelah ffill.")
    
    # 7. Deteksi Lonjakan Harga (Jump Flag)
    print("📈 Mendeteksi lonjakan harga harian (>5%)...")
    df_merged['return_harian'] = df_merged['proksi_antam_idr'].pct_change()
    df_merged['is_jump'] = (df_merged['return_harian'].abs() > config.JUMP_FLAG).astype(int)
    
    # 8. Simpan Hasil
    output_path = config.PROC / "harian.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"\n✅ BERHASIL! Data harian disimpan di: {output_path}")
    print(f"📊 Total baris akhir: {len(df_merged)}")
    print(f"📆 Rentang tanggal: {df_merged['tanggal'].min()} s.d. {df_merged['tanggal'].max()}")
    
    # Simpan laporan lonjakan untuk diperiksa manual
    jump_df = df_merged[df_merged['is_jump'] == 1][['tanggal', 'proksi_antam_idr', 'return_harian']]
    jump_path = config.OUT / "lompatan_ditandai.csv"
    jump_df.to_csv(jump_path, index=False)
    print(f"🚨 Ditemukan {len(jump_df)} hari dengan lonjakan >{config.JUMP_FLAG*100}%. Cek: {jump_path}")

if __name__ == "__main__":
    main()