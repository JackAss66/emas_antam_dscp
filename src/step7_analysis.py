import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import config
import warnings

warnings.filterwarnings('ignore')

def diebold_mariano_test(e1, e2):
    """
    Uji Diebold-Mariano untuk membandingkan akurasi dua model.
    e1: Error model baseline (M0)
    e2: Error model canggih (M2/M3)
    """
    d = e1**2 - e2**2
    T = len(d)
    mean_d = np.mean(d)
    
    # Varians dengan koreksi autokorelasi sederhana (Newey-West untuk lag 1)
    var_d = np.var(d, ddof=1)
    if T > 2:
        gamma1 = np.cov(d[:-1], d[1:])[0, 1]
        if not np.isnan(gamma1):
            var_d = var_d + 2 * gamma1 * (1 - 1/T)
            
    se_d = np.sqrt(var_d / T)
    if se_d == 0 or np.isnan(se_d):
        return 0.0, 1.0
        
    dm_stat = mean_d / se_d
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def load_averaged_preds(prefix):
    """Memuat dan merata-ratakan prediksi dari 5 seed (M2_Seed0 s.d. M2_Seed4)."""
    files = list(config.PRED_DIR.glob(f"{prefix}_Seed*.csv"))
    if not files:
        return None
        
    # PERBAIKAN: Tambahkan parse_dates=['tanggal'] agar formatnya sama dengan df_fitur
    dfs = [pd.read_csv(f, parse_dates=['tanggal']) for f in files]
    
    df_avg = dfs[0][['tanggal', 'blok', 'y_true']].copy()
    # Rata-ratakan y_pred dari semua seed
    df_avg['y_pred'] = np.mean([df['y_pred'].values for df in dfs], axis=0)
    return df_avg

def main():
    print("🚀 MEMULAI TAHAP 7: ANALISIS LANJUTAN (Uji Statistik, Volatilitas, & Error)...")
    print("="*70)
    
    # 1. Muat Data
    df_fitur = pd.read_csv(config.PROC / "fitur.csv", parse_dates=['tanggal'])
    df_m0 = pd.read_csv(config.PRED_DIR / "M0_Naive.csv", parse_dates=['tanggal'])
    df_m2 = load_averaged_preds("M2") # LSTM tanpa sentimen
    df_m3 = load_averaged_preds("M3") # LSTM dengan sentimen
    
    if df_m2 is None or df_m3 is None:
        print("❌ Error: File prediksi M2 atau M3 (Seed 0-4) tidak ditemukan. Pastikan Step 5/6 sudah selesai.")
        return

    # Merge data untuk analisis
    df_m0 = pd.merge(df_m0, df_fitur[['tanggal', 'skor_bersih']], on='tanggal', how='left')
    df_m2 = pd.merge(df_m2, df_fitur[['tanggal', 'skor_bersih']], on='tanggal', how='left')
    df_m3 = pd.merge(df_m3, df_fitur[['tanggal', 'skor_bersih']], on='tanggal', how='left')

    # Hitung Error (Abs)
    df_m0['error'] = np.abs(df_m0['y_true'] - df_m0['y_pred'])
    df_m2['error'] = np.abs(df_m2['y_true'] - df_m2['y_pred'])
    df_m3['error'] = np.abs(df_m3['y_true'] - df_m3['y_pred'])

    # =========================================================================
    # BAGIAN 1: UJI DIEBOLD-MARIANO (Signifikansi Statistik)
    # =========================================================================
    print("\n📊 1. UJI DIEBOLD-MARIANO (DM TEST)")
    print("Menguji apakah penurunan error secara statistik signifikan (H0: Akurasi sama).")
    
    # M2 vs M0
    dm_20, p_20 = diebold_mariano_test(df_m0['y_true'] - df_m0['y_pred'], df_m2['y_true'] - df_m2['y_pred'])
    print(f"   • M2 (LSTM) vs M0 (Naive): DM Stat = {dm_20:.4f}, p-value = {p_20:.4f}")
    if p_20 < 0.05 and dm_20 > 0:
        print("     ✅ KESIMPULAN: M2 SIGNIFIKAN LEBIH BAIK dari M0 (p < 0.05).")
    else:
        print("     ⚠️ KESIMPULAN: Tidak ada perbedaan signifikan antara M2 dan M0 (p >= 0.05).")

    # M3 vs M2
    dm_32, p_32 = diebold_mariano_test(df_m2['y_true'] - df_m2['y_pred'], df_m3['y_true'] - df_m3['y_pred'])
    print(f"   • M3 (LSTM+Sentimen) vs M2 (LSTM): DM Stat = {dm_32:.4f}, p-value = {p_32:.4f}")
    if p_32 < 0.05 and dm_32 > 0:
        print("     ✅ KESIMPULAN: Penambahan Sentimen SIGNIFIKAN meningkatkan akurasi (p < 0.05).")
    else:
        print("     ⚠️ KESIMPULAN: Penambahan fitur sentimen TIDAK memberikan peningkatan signifikan secara statistik.")

    # =========================================================================
    # BAGIAN 2: ANALISIS REGIME PASAR (Tenang vs Bergejolak)
    # =========================================================================
    print("\n📊 2. ANALISIS KINERJA BERDASARKAN VOLATILITAS PASAR")
    
    # Hitung volatilitas rolling 20 hari
    df_fitur['vol_20d'] = df_fitur['target_return'].rolling(window=20).std()
    threshold = df_fitur['vol_20d'].quantile(0.75) # 75th percentile sebagai batas gejolak
    df_fitur['regime'] = np.where(df_fitur['vol_20d'] >= threshold, 'Bergejolak', 'Tenang')
    
    # Merge regime ke data prediksi
    df_m2 = pd.merge(df_m2, df_fitur[['tanggal', 'regime']], on='tanggal', how='left').dropna(subset=['regime'])
    df_m3 = pd.merge(df_m3, df_fitur[['tanggal', 'regime']], on='tanggal', how='left').dropna(subset=['regime'])
    
    for regime in ['Tenang', 'Bergejolak']:
        rmse_m2 = np.sqrt(np.mean((df_m2[df_m2['regime']==regime]['y_true'] - df_m2[df_m2['regime']==regime]['y_pred'])**2))
        rmse_m3 = np.sqrt(np.mean((df_m3[df_m3['regime']==regime]['y_true'] - df_m3[df_m3['regime']==regime]['y_pred'])**2))
        print(f"   • Periode {regime.upper()}: RMSE M2 = {rmse_m2:.6f} | RMSE M3 = {rmse_m3:.6f}")
        if rmse_m3 < rmse_m2:
            print(f"     👉 Sentimen membantu mengurangi error pada kondisi {regime}.")
        else:
            print(f"     👉 Sentimen tidak membantu (atau memperburuk) pada kondisi {regime}.")

    # =========================================================================
    # BAGIAN 3: BEDAH 10 ERROR TERBESAR (Top Errors Analysis)
    # =========================================================================
    print("\n📊 3. ANALISIS KUALITATIF: 10 HARI DENGAN ERROR PREDIKSI TERBESAR (Model M3)")
    top_errors = df_m3.nlargest(10, 'error')[['tanggal', 'y_true', 'y_pred', 'error', 'skor_bersih', 'blok']]
    
    print(f"{'Tanggal':<12} | {'Return Asli':<12} | {'Prediksi':<12} | {'Error':<10} | {'Skor Sentimen':<12} | Blok")
    print("-" * 85)
    for _, row in top_errors.iterrows():
        tgl = pd.to_datetime(row['tanggal']).strftime('%Y-%m-%d')
        print(f"{tgl:<12} | {row['y_true']:>11.5f} | {row['y_pred']:>11.5f} | {row['error']:>9.5f} | {row['skor_bersih']:>11.3f}      | {int(row['blok'])}")

    print("\n TIP UNTUK BAB 4 SKRIPSI:")
    print("   - Gunakan hasil DM Test untuk menjawab hipotesis penelitian secara statistik.")
    print("   - Gunakan analisis regime untuk membahas 'Keterbatasan Model' (misal: model gagal saat ada berita hitam/kejutan).")
    print("   - Analisis Top Error menunjukkan bahwa model kesulitan memprediksi 'outlier' yang tidak terwakili dalam pola historis.")
    
    # Simpan hasil analisis ke CSV untuk lampiran
    top_errors.to_csv(config.OUT / "analysis_top_errors.csv", index=False)
    print(f"\n✅ Detail Top Error disimpan di: {config.OUT / 'analysis_top_errors.csv'}")

if __name__ == "__main__":
    main()