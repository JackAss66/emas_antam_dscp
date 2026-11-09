import pandas as pd
import numpy as np
from pathlib import Path
import config
import warnings

warnings.filterwarnings('ignore')

def main():
    print("🚀 MEMULAI TAHAP 4: PEMBUATAN FITUR DAN PEMBAGIAN WALK-FORWARD...\n")
    
    # 1. Muat Data
    print("📥 Memuat data harian dan sentimen...")
    df_harian = pd.read_csv(config.PROC / "harian.csv", parse_dates=['tanggal'])
    df_sentimen = pd.read_csv(config.PROC / "sentimen_harian.csv", parse_dates=['tanggal_pakai'])
    
    # 2. Merge Data
    # Kita merge berdasarkan tanggal_harian == tanggal_pakai
    # Artinya, fitur sentimen dari berita hari D-1 akan muncul di baris hari D
    df = pd.merge(df_harian, df_sentimen, left_on='tanggal', right_on='tanggal_pakai', how='left')
    
    # Urutkan berdasarkan tanggal
    df = df.sort_values('tanggal').reset_index(drop=True)
    
    # 3. Pembersihan Fitur Sentimen
    # Jika tidak ada berita pada hari tersebut, skor_bersih diisi 0.0 (netral)
    df['skor_bersih'] = df['skor_bersih'].fillna(0.0)
    df['ada_berita'] = df['ada_berita'].fillna(0).astype(int)
    
    # 4. Pembuatan Target dan Fitur Return (PENTING: Hindari Data Leakage)
    # Target: Return harian pada hari yang sama (D). 
    df['target_return'] = df['proksi_antam_idr'].pct_change()
    
    # Fitur: Return dari hari sebelumnya (D-1). 
    # Ini adalah informasi yang sah diketahui di awal hari D untuk memprediksi hari D.
    df['lag_return_antam'] = df['target_return'].shift(1)
    df['lag_return_emas_usd'] = df['emas_usd_close'].pct_change().shift(1)
    df['lag_return_kurs'] = df['kurs_idr'].pct_change().shift(1)
    
    # Hapus baris pertama yang NaN karena operasi pct_change() dan shift()
    df = df.dropna(subset=['target_return', 'lag_return_antam', 'lag_return_emas_usd', 'lag_return_kurs']).reset_index(drop=True)
    
    # 5. Definisi Set Fitur
    # Set A: Hanya data harga historis (Baseline untuk LSTM)
    FEATURES_A = ['lag_return_antam', 'lag_return_emas_usd', 'lag_return_kurs']
    # Set B: Data harga historis + Sentimen
    FEATURES_B = FEATURES_A + ['skor_bersih']
    
    print(f"✅ Fitur Set A (Tanpa Sentimen): {FEATURES_A}")
    print(f"✅ Fitur Set B (Dengan Sentimen): {FEATURES_B}")
    
    # 6. Pembagian Walk-Forward (Expanding Window)
    print(f"\n📅 Membagi data menjadi {config.N_BLOCKS} blok walk-forward...")
    
    total_days = len(df)
    # Strategi: Gunakan 60% data awal untuk training dasar yang cukup besar, 
    # dan 40% sisanya dibagi rata menjadi N_BLOCKS untuk pengujian.
    train_initial_size = int(total_days * 0.60)
    remaining_days = total_days - train_initial_size
    test_size_per_block = remaining_days // config.N_BLOCKS
    
    blocks = []
    current_train_end = train_initial_size
    
    for i in range(1, config.N_BLOCKS + 1):
        train_start = 0
        train_end = current_train_end
        test_start = current_train_end
        test_end = current_train_end + test_size_per_block
        
        # Blok terakhir mencakup sisa data hingga akhir
        if i == config.N_BLOCKS:
            test_end = total_days
            
        blocks.append({
            'blok': i,
            'train_start_idx': train_start,
            'train_end_idx': train_end - 1,
            'test_start_idx': test_start,
            'test_end_idx': test_end - 1,
            'train_start_date': df.iloc[train_start]['tanggal'].date(),
            'train_end_date': df.iloc[train_end - 1]['tanggal'].date(),
            'test_start_date': df.iloc[test_start]['tanggal'].date(),
            'test_end_date': df.iloc[test_end - 1]['tanggal'].date(),
            'train_days': train_end - train_start,
            'test_days': test_end - test_start
        })
        
        # Expanding window: akhir periode train berikutnya bergeser sejauh periode test yang baru saja selesai
        current_train_end = test_end
        
    df_blocks = pd.DataFrame(blocks)
    
    # 7. Simpan Hasil
    fitur_path = config.PROC / "fitur.csv"
    df.to_csv(fitur_path, index=False)
    print(f"\n💾 Data fitur final disimpan di: {fitur_path}")
    print(f"   Total baris siap pemodelan: {len(df)}")
    
    blok_path = config.PROC / "blok_walkforward.csv"
    df_blocks.to_csv(blok_path, index=False)
    print(f"💾 Skema walk-forward disimpan di: {blok_path}")
    
    print("\n📊 RINGKASAN SKEMA WALK-FORWARD:")
    print(df_blocks[['blok', 'train_days', 'test_days', 'train_start_date', 'test_end_date']].to_string(index=False))
    
    # 8. Pemeriksaan Kebocoran Data (Sanity Check)
    print("\n🔒 PEMERIKSAAN KEBOCORAN DATA:")
    # Pastikan tidak ada tanggal test yang <= tanggal train di blok manapun
    leakage = (df_blocks['test_start_date'] <= df_blocks['train_end_date']).any()
    if not leakage:
        print("   [AMAN] Tidak ada tumpang tindih tanggal antara train dan test di semua blok.")
    else:
        print("   [BAHAYA] Ditemukan tumpang tindih tanggal! Periksa logika pembagian.")

if __name__ == "__main__":
    main()