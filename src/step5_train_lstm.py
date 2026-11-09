import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import config
import json
import argparse
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def train_model(X_tr, y_tr, X_va, y_va, X_te, params, seed, epochs=50, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va))
    test_ds = TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(np.zeros(len(X_te))))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = LSTMModel(input_size=X_tr.shape[2], hidden_size=params['units'], dropout=params['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss, patience_counter, best_state = float('inf'), 0, None
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward(); optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_state = model.state_dict(); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
                
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            preds.extend(model(X_b.to(device)).cpu().numpy().flatten())
            
    # PERBAIKAN: Kembalikan prediksi DAN loss validasi
    return np.array(preds), best_val_loss 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True, choices=['A', 'B'])
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    print(f"🚀 TAHAP 5B: LSTM (Set {args.features}, Seed {args.seed})\n")
    df = pd.read_csv(config.PROC / "fitur.csv", parse_dates=['tanggal'])
    df_blocks = pd.read_csv(config.PROC / "blok_walkforward.csv")
    
    feat_cols = config.FEATURES_A if args.features == 'A' else config.FEATURES_B
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    all_preds = []
    best_params_file = config.OUT / f"best_params_{args.features}.json"
    
    for _, block in df_blocks.iterrows():
        blok = int(block['blok'])
        tr_end = int(block['train_end_idx'])
        te_start = int(block['test_start_idx'])
        te_end = int(block['test_end_idx'])
        
        df_b = df.iloc[:te_end+1].copy()
        X_all = df_b[feat_cols].values
        y_all = df_b['target_return'].values
        
        X_tr_val = X_all[:tr_end+1]; y_tr_val = y_all[:tr_end+1]
        X_te = X_all[te_start:te_end+1]; y_te = y_all[te_start:te_end+1]
        dates_te = df_b['tanggal'].iloc[te_start:te_end+1].values
        
        sc_X = StandardScaler(); sc_y = StandardScaler()
        X_tr_val_sc = sc_X.fit_transform(X_tr_val)
        y_tr_val_sc = sc_y.fit_transform(y_tr_val.reshape(-1, 1)).flatten()
        X_te_sc = sc_X.transform(X_te)
        
        if args.seed == 0:
            print(f"  🔍 Blok {blok}: Grid Search...")
            val_size = max(10, int(len(X_tr_val_sc) * config.VAL_FRAC))
            X_tr = X_tr_val_sc[:-val_size]; y_tr = y_tr_val_sc[:-val_size]
            X_va = X_tr_val_sc[-val_size:]; y_va = y_tr_val_sc[-val_size:]
            
            best_vl, best_p = float('inf'), None
            for w in config.WINDOWS:
                for u in config.UNITS:
                    for d in config.DROPOUTS:
                        X_ts, y_ts = create_sequences(X_tr, y_tr, w)
                        X_vs, y_vs = create_sequences(X_va, y_va, w)
                        if len(X_ts) < 10: continue
                        
                        _, vl = train_model(X_ts, y_ts, X_vs, y_vs, X_vs, {'units': u, 'dropout': d}, args.seed, epochs=30, patience=5)
                        if vl < best_vl: best_vl = vl; best_p = {'window': w, 'units': u, 'dropout': d}
            print(f"  ✅ Best Params: {best_p}")
            with open(best_params_file, 'w') as f: json.dump(best_p, f)
        else:
            with open(best_params_file, 'r') as f: best_p = json.load(f)
            print(f"  📂 Blok {blok}: Load Params {best_p}")
            
        w = best_p['window']
        
        # PERBAIKAN: Buat sequence 3D dari data train+val yang sudah di-scale
        X_tr_val_seq, y_tr_val_seq = create_sequences(X_tr_val_sc, y_tr_val_sc, w)
        
        # Trik penting: gabungkan train+val dan test untuk membuat sequence yang kontinu untuk test set
        X_combined = np.vstack((X_tr_val_sc, X_te_sc))
        X_combined_seq, _ = create_sequences(X_combined, np.zeros(len(X_combined)), w)
        te_seq_start = len(X_tr_val_sc) - w
        X_te_seq = X_combined_seq[te_seq_start : te_seq_start + len(X_te)]

        # Split data train dan validasi akhir
        val_size_final = max(10, int(len(X_tr_val_seq) * 0.1))
        X_tr = X_tr_val_seq[:-val_size_final]
        y_tr = y_tr_val_seq[:-val_size_final]
        X_va = X_tr_val_seq[-val_size_final:]
        y_va = y_tr_val_seq[-val_size_final:]

        # Latih model final dan dapatkan prediksi
        final_preds_scaled, _ = train_model(X_tr, y_tr, X_va, y_va, X_te_seq, best_p, args.seed, epochs=100, patience=15)
        
        # Kembalikan ke skala asli
        final_preds = sc_y.inverse_transform(final_preds_scaled.reshape(-1, 1)).flatten()
        
        # Inverse transform untuk mendapatkan prediksi dalam skala asli
        final_preds = sc_y.inverse_transform(final_preds_scaled.reshape(-1, 1)).flatten()
        
        pred_df = pd.DataFrame({
            'tanggal': dates_te, 'blok': blok, 
            'model': f"M{'2' if args.features == 'A' else '3'}_LSTM", 
            'seed': args.seed, 'y_true': y_te, 'y_pred': final_preds
        })
        all_preds.append(pred_df)
        
    final = pd.concat(all_preds, ignore_index=True)
    out_name = f"M{'2' if args.features == 'A' else '3'}_Seed{args.seed}.csv"
    final.to_csv(config.PRED_DIR / out_name, index=False)
    print(f"✅ Selesai! Disimpan di: {config.PRED_DIR / out_name}\n")

if __name__ == "__main__":
    main()