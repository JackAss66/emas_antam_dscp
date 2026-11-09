import pandas as pd
import numpy as np
from pathlib import Path
import config
import warnings

warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    """Menghitung RMSE, MAE, dan Directional Accuracy."""
    # 1. RMSE & MAE (berdasarkan Return)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 2. Directional Accuracy (DA)
    # Kita hanya evaluasi arah pada hari di mana harga benar-benar bergerak (|return| > 0.0001)
    # untuk menghindari bias dari hari yang datar (flat).
    mask_bergerak = np.abs(y_true) > 1e-4 
    
    if mask_bergerak.sum() > 0:
        # Bandingkan tanda (sign) antara asli dan prediksi
        benar = np.sum(np.sign(y_true[mask_bergerak]) == np.sign(y_pred[mask_bergerak]))
        total = mask_bergerak.sum()
        da = benar / total
    else:
        da = np.nan
        
    return rmse, mae, da

def main():
    print(" MELAKUKAN EVALUASI MODEL (STEP 6)...")
    print("="*50)
    
    # Cari semua file prediksi di folder outputs/preds/
    pred_files = list(config.PRED_DIR.glob("*.csv"))
    
    if not pred_files:
        print("❌ Tidak ada file prediksi ditemukan di outputs/preds/. Pastikan Step 5 (Training) sudah selesai.")
        return
        
    all_results = []
    
    # Loop setiap file prediksi
    for p_file in pred_files:
        model_name = p_file.stem  # Contoh: M2_Seed0, M3_Seed1
        df_pred = pd.read_csv(p_file)
        
        # Pastikan kolom yang dibutuhkan ada
        if 'y_true' not in df_pred.columns or 'y_pred' not in df_pred.columns:
            print(f"️ Melewatkan {model_name} (format tidak sesuai)")
            continue
            
        y_true = df_pred['y_true'].values
        y_pred = df_pred['y_pred'].values
        
        # Hitung metrik untuk seluruh periode (gabungan semua blok)
        rmse, mae, da = calculate_metrics(y_true, y_pred)
        
        # Hitung metrik per blok (untuk melihat stabilitas model)
        for blok in df_pred['blok'].unique():
            df_blok = df_pred[df_pred['blok'] == blok]
            r_b, m_b, d_b = calculate_metrics(df_blok['y_true'].values, df_blok['y_pred'].values)
            
            all_results.append({
                'model_base': model_name.split('_Seed')[0], # M0, M1, M2, M3
                'full_name': model_name,
                'blok': blok,
                'rmse': r_b,
                'mae': m_b,
                'directional_accuracy': d_b,
                'n_samples': len(df_blok)
            })
            
    df_results = pd.DataFrame(all_results)
    
    # Simpan hasil detail per blok
    df_results.to_csv(config.OUT / "results_detail_per_blok.csv", index=False)
    
    # Buat ringkasan rata-rata per Model (M0, M1, M2, M3)
    df_summary = df_results.groupby('model_base').agg({
        'rmse': 'mean',
        'mae': 'mean',
        'directional_accuracy': 'mean',
        'n_samples': 'sum'
    }).reset_index()
    
    # Urutkan berdasarkan Directional Accuracy (tertinggi ke terendah)
    df_summary = df_summary.sort_values('directional_accuracy', ascending=False)
    
    df_summary.to_csv(config.OUT / "results_summary.csv", index=False)
    
    print("\n RINGKASAN KINERJA MODEL (Rata-rata dari semua Blok & Seed):")
    print(df_summary.to_string(index=False))
    print("\n✅ Hasil evaluasi disimpan di folder outputs/")
    print("   - results_summary.csv (Untuk perbandingan utama)")
    print("   - results_detail_per_blok.csv (Untuk analisis stabilitas)")

if __name__ == "__main__":
    main()