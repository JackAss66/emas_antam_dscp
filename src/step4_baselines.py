import pandas as pd
import numpy as np
from pathlib import Path
import config

def main():
    print("🚀 MEMULAI TAHAP 5A: PEMODELAN BASELINE (M0 & M1)...\n")
    
    df = pd.read_csv(config.PROC / "fitur.csv", parse_dates=['tanggal'])
    df_blocks = pd.read_csv(config.PROC / "blok_walkforward.csv")
    
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    
    all_preds_m0 = []
    all_preds_m1 = []
    
    for _, block in df_blocks.iterrows():
        blok_num = int(block['blok'])
        train_end_idx = int(block['train_end_idx'])
        test_start_idx = int(block['test_start_idx'])
        test_end_idx = int(block['test_end_idx'])
        
        df_train = df.iloc[:train_end_idx+1].copy()
        df_test = df.iloc[test_start_idx:test_end_idx+1].copy()
        
        # M0: Naive (Prediksi Return = 0.0)
        df_test['y_pred_m0'] = 0.0
        
        # M1: Mean Training Return
        mean_train_return = df_train['target_return'].mean()
        if pd.isna(mean_train_return): mean_train_return = 0.0
        df_test['y_pred_m1'] = mean_train_return
        
        pred_m0 = df_test[['tanggal', 'target_return', 'y_pred_m0']].copy()
        pred_m0['model'] = 'M0_Naive'
        pred_m0['blok'] = blok_num
        
        pred_m1 = df_test[['tanggal', 'target_return', 'y_pred_m1']].copy()
        pred_m1['model'] = 'M1_MeanTrain'
        pred_m1['blok'] = blok_num
        
        all_preds_m0.append(pred_m0)
        all_preds_m1.append(pred_m1)
        
    final_m0 = pd.concat(all_preds_m0, ignore_index=True).rename(columns={'target_return': 'y_true', 'y_pred_m0': 'y_pred'})
    final_m1 = pd.concat(all_preds_m1, ignore_index=True).rename(columns={'target_return': 'y_true', 'y_pred_m1': 'y_pred'})
    
    final_m0.to_csv(config.PRED_DIR / "M0_Naive.csv", index=False)
    final_m1.to_csv(config.PRED_DIR / "M1_MeanTrain.csv", index=False)
    
    print("✅ BERHASIL! Prediksi baseline disimpan di outputs/preds/")

if __name__ == "__main__":
    main()