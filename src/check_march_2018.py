import pandas as pd
from pathlib import Path
import config

def check_march_2018():
    print("🔍 MEMERIKSA DATA HARIAN & SENTIMEN: 1 - 10 MARET 2018\n")
    
    # 1. Cek Data Harga
    df_harian = pd.read_csv(config.PROC / "harian.csv", parse_dates=['tanggal'])
    mask_harian = (df_harian['tanggal'] >= '2018-03-01') & (df_harian['tanggal'] <= '2018-03-10')
    df_march = df_harian[mask_harian].copy()
    
    print("📊 PERGERAKAN HARGA (Proksi Antam & Kurs):")
    cols_to_show = ['tanggal', 'proksi_antam_idr', 'kurs_idr', 'return_harian', 'is_jump']
    print(df_march[cols_to_show].to_string(index=False))
    print("-" * 80)

    # 2. Cek Data Sentimen Berita
    sentimen_path = config.PROC / "sentimen_harian.csv"
    if sentimen_path.exists():
        df_sentimen = pd.read_csv(sentimen_path, parse_dates=['tanggal_pakai'])
        # Kita cari berita yang diterbitkan sekitar 28 Feb - 9 Maret (karena dipakai untuk prediksi D+1)
        mask_berita = (df_sentimen['tanggal_berita'] >= '2018-02-28') & (df_sentimen['tanggal_berita'] <= '2018-03-09')
        df_berita_march = df_sentimen[mask_berita].copy()
        
        print("📰 SKOR SENTIMEN BERITA (Tanggal Pakai = Tanggal Prediksi):")
        cols_berita = ['tanggal_berita', 'tanggal_pakai', 'n_berita', 'n_naik', 'n_turun', 'skor_bersih']
        print(df_berita_march[cols_berita].to_string(index=False))
    else:
        print("⚠️ File sentimen_harian.csv belum dibuat. Jalankan step3_sentiment.py terlebih dahulu.")

if __name__ == "__main__":
    check_march_2018()