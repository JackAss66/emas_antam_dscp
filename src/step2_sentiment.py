import pandas as pd
import numpy as np
import re
from pathlib import Path
import config

def clean_yfinance(df, price_col_name):
    """Membersihkan format CSV yfinance yang memiliki baris metadata di awal."""
    df_clean = df.iloc[2:].copy() # Skip 2 baris metadata
    df_clean.columns = ['Date', price_col_name] + list(df.columns[2:])
    df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.date
    df_clean = df_clean[['Date', price_col_name]]
    df_clean[price_col_name] = pd.to_numeric(df_clean[price_col_name], errors='coerce')
    return df_clean

def extract_harga(text):
    """Fungsi robust untuk mengekstrak angka dari string seperti '1.234.567( +100 )'"""
    if pd.isna(text):
        return np.nan
    match = re.search(r'[\d\.]+', str(text))
    if match:
        return float(match.group().replace('.', ''))
    return np.nan

def main():
    print("🚀 MEMULAI PEMBERSIHAN DAN PENGGABUNGAN DATA HARIAN (v2 - Robust)...\n")
    
    # 1. Baca Data Emas Dunia (GC=F)
    print("📥 Membaca data GC=F (yfinance)...")
    df_antam = pd.read_csv(config.ANTAM_CSV)
    df_antam = clean_yfinance(df_antam, config.ANTAM_PRICE_COL)
    df_antam.rename(columns={'Date': 'tanggal', config.ANTAM_PRICE_COL: 'emas_usd_close'}, inplace=True)
    
    # 2. Baca Data Kurs
    print("📥 Membaca data Kurs USD/IDR (yfinance)...")
    df_kurs = pd.read_csv(config.KURS_CSV)
    df_kurs = clean_yfinance(df_kurs, config.KURS_PRICE_COL)
    df_kurs.rename(columns={'Date': 'tanggal', config.KURS_PRICE_COL: 'kurs_idr'}, inplace=True)
    
    # 3. Baca Data Emas Dunia (Net) & Ekstraksi Proksi Antam
    print("📥 Membaca dan membersihkan data Emas Net (Proksi Antam)...")
    df_emas_net = pd.read_csv(config.EMAS_DUNIA_CSV)
    df_emas_net['Tanggal'] = pd.to_datetime(df_emas_net['Tanggal'], dayfirst=True, errors='coerce').dt.date
    
    # Filter hanya satuan Gram
    df_emas_gram = df_emas_net[df_emas_net['Satuan'] == 'Gram(gr)'].copy()
    
    # TERAPKAN FUNGSI EKSTRAKSI YANG ROBUST
    df_emas_gram['proksi_antam_idr'] = df_emas_gram['Kurs Tengah BI'].apply(extract_harga)
    df_emas_gram = df_emas_gram[['Tanggal', 'proksi_antam_idr']].rename(columns={'Tanggal': 'tanggal'})
    
    # 4. Merge Semua Data
    print("🔗 Menggabungkan data berdasarkan tanggal...")
    df_merged = df_antam.merge(df_kurs, on='tanggal', how='outer')
    df_merged = df_merged.merge(df_emas_gram, on='tanggal', how='outer')
    
    # 5. Filter Periode Penelitian
    start_date = pd.to_datetime(config.START_DATE).date()
    end_date = pd.to_datetime(config.END_DATE).date()
    df_merged = df_merged[(df_merged['tanggal'] >= start_date) & (df_merged['tanggal'] <= end_date)]
    df_merged = df_merged.sort_values('tanggal').reset_index(drop=True)
    
    # 6. Pembersihan Lanjutan: Ganti 0 dengan NaN, lalu Interpolasi Linier
    print("🧹 Membersihkan anomali (mengganti 0/NaN dengan interpolasi linier)...")
    cols_to_clean = ['emas_usd_close', 'kurs_idr', 'proksi_antam_idr']
    
    for col in cols_to_clean:
        # Tandai baris yang asli-nya NaN atau 0
        df_merged[f'is_interpolated_{col}'] = ((df_merged[col].isna()) | (df_merged[col] == 0)).astype(int)
        
        # Ganti 0 menjadi NaN agar bisa diinterpolasi
        df_merged[col] = df_merged[col].replace(0, np.nan)
        
        # Interpolasi linier (jika NaN di awal/akhir, baru ffill/bfill)
        df_merged[col] = df_merged[col].interpolate(method='linear').bfill().ffill()

    # 7. Deteksi Lonjakan Harga (Jump Flag)
    print("📈 Mendeteksi lonjakan harga harian (>5%)...")
    df_merged['return_harian'] = df_merged['proksi_antam_idr'].pct_change()
    df_merged['is_jump'] = (df_merged['return_harian'].abs() > config.JUMP_FLAG).astype(int)
    
    # 8. Simpan Hasil
    output_path = config.PROC / "harian.csv"
    df_merged.to_csv(output_path, index=False)
    print(f"\n✅ BERHASIL! Data harian disimpan di: {output_path}")
    print(f"📊 Total baris akhir: {len(df_merged)}")
    
    # Simpan laporan lonjakan
    jump_df = df_merged[df_merged['is_jump'] == 1][['tanggal', 'proksi_antam_idr', 'return_harian']]
    jump_path = config.OUT / "lompatan_ditandai.csv"
    jump_df.to_csv(jump_path, index=False)
    print(f"🚨 Ditemukan {len(jump_df)} hari dengan lonjakan >{config.JUMP_FLAG*100}%. Cek: {jump_path}")

if __name__ == "__main__":
    main()