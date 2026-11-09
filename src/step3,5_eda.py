import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config
import warnings

warnings.filterwarnings('ignore')

# Pastikan folder figures ada
config.FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 MEMULAI TAHAP 3.5: EXPLORATORY DATA ANALYSIS (EDA)...\n")
    
    # 1. Muat Data Fitur Final (dari step4)
    print("📥 Memuat data fitur final...")
    df = pd.read_csv(config.PROC / "fitur.csv", parse_dates=['tanggal'])
    df = df.sort_values('tanggal').reset_index(drop=True)
    
    # =========================================================================
    # VISUALISASI 1: Tren Harga Proksi Antam & Kurs USD/IDR
    # =========================================================================
    print("📊 Membuat grafik tren harga dan kurs...")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Tanggal')
    ax1.set_ylabel('Harga Proksi Antam (IDR/gram)', color=color)
    ax1.plot(df['tanggal'], df['proksi_antam_idr'], color=color, label='Harga Proksi Antam', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Kurs USD/IDR', color=color)
    ax2.plot(df['tanggal'], df['kurs_idr'], color=color, linestyle='--', label='Kurs USD/IDR', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Tren Harga Emas Proksi dan Kurs USD/IDR (2015-2019)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(config.FIG_DIR / "01_tren_harga_dan_kurs.png", dpi=300)
    plt.close()

    # =========================================================================
    # VISUALISASI 2: Distribusi Return Harian (Cek Fat Tails)
    # =========================================================================
    print(" Membuat histogram distribusi return harian...")
    plt.figure(figsize=(10, 6))
    # Hapus outlier ekstrem agar histogram terlihat jelas (opsional, tapi disarankan untuk visualisasi)
    returns_clean = df['target_return'].clip(-0.05, 0.05)
    sns.histplot(returns_clean, bins=50, kde=True, color='purple', stat='density')
    plt.title('Distribusi Return Harian Harga Emas Proksi (Dipotong ±5%)', fontsize=14)
    plt.xlabel('Return Harian')
    plt.ylabel('Densitas')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "02_distribusi_return.png", dpi=300)
    plt.close()

    # =========================================================================
    # VISUALISASI 3: Matriks Korelasi (Heatmap) - KRITIS UNTUK SKRIPSI
    # =========================================================================
    print("📊 Membuat heatmap korelasi antar fitur...")
    # Pilih kolom yang relevan untuk korelasi
    cols_to_corr = ['target_return', 'lag_return_antam', 'lag_return_emas_usd', 'lag_return_kurs', 'skor_bersih']
    df_corr = df[cols_to_corr].corr()
    
    plt.figure(figsize=(8, 6))
    # Gunakan annot=True agar angka korelasi terlihat
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, center=0)
    plt.title('Matriks Korelasi: Target vs Fitur (Termasuk Sentimen)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "03_korelasi_fitur.png", dpi=300)
    plt.close()

    # =========================================================================
    # VISUALISASI 4: Volatilitas Pasar (Rolling Standard Deviation)
    # =========================================================================
    print("📊 Menghitung dan memplot volatilitas pasar (Rolling 20 hari)...")
    df['volatilitas_20d'] = df['target_return'].rolling(window=config.VOL_WINDOW).std()
    
    plt.figure(figsize=(14, 6))
    plt.plot(df['tanggal'], df['volatilitas_20d'], color='orange', label=f'Volatilitas ({config.VOL_WINDOW} hari)', linewidth=1.5)
    
    # Tandai periode bergejolak (di atas persentil ke-75)
    threshold = df['volatilitas_20d'].quantile(config.VOL_PERCENTILE / 100)
    plt.axhline(threshold, color='red', linestyle='--', label=f'Batas Gejolak (Persentil {config.VOL_PERCENTILE})')
    plt.fill_between(df['tanggal'], df['volatilitas_20d'], threshold, 
                     where=(df['volatilitas_20d'] >= threshold), color='red', alpha=0.3, label='Periode Bergejolak')
    
    plt.title('Volatilitas Pasar Emas (Rolling Standard Deviation)', fontsize=14, fontweight='bold')
    plt.xlabel('Tanggal')
    plt.ylabel('Standar Deviasi Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "04_volatilitas_pasar.png", dpi=300)
    plt.close()

    # =========================================================================
    # VISUALISASI 5: Tren Skor Sentimen Berita
    # =========================================================================
    print("📊 Membuat grafik tren skor sentimen...")
    plt.figure(figsize=(14, 6))
    plt.bar(df['tanggal'], df['skor_bersih'], color='gray', alpha=0.7, label='Skor Sentimen Bersih', width=1.0)
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Tren Skor Sentimen Berita Harian (Net Sentiment Score)', fontsize=14, fontweight='bold')
    plt.xlabel('Tanggal')
    plt.ylabel('Skor Bersih (-1 s.d. 1)')
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "05_tren_sentimen.png", dpi=300)
    plt.close()

    print("\n✅ EDA SELESAI! Semua grafik disimpan di folder: outputs/figures/")
    print("💡 TIP UNTUK SKRIPSI:")
    print("   - Gunakan Gambar 03 untuk membuktikan apakah sentimen berkorelasi dengan return.")
    print("   - Gunakan Gambar 04 sebagai dasar pembagian 'Periode Tenang' vs 'Bergejolak' di Bab Evaluasi.")

if __name__ == "__main__":
    main()