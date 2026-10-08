"""Run seed, temporal-split, and latent-dimension robustness experiments.

The script reuses the prepared training tensors and evaluates the same CVAE
under alternative random seeds, validation months, and latent dimensions.
Results and trained comparison weights are written to local experiment
directories.
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import os
import sys
import random
import model as M
from train import load_data, prepare_cvae_data_emb  # reuse your existing functions

sys.stdout.reconfigure(line_buffering=True)
sys.stderr = sys.stdout

path = '../'
os.makedirs(path + "models/robustness/", exist_ok=True)

# --- Best hyperparameters from Trial 42 ---
BEST_LR      = 0.0003
BEST_WD      = 0.0001
BEST_DZ      = 15
BEST_BETA    = 1
BEST_HIDDEN  = [256]  # Nl=2, Nn=64

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(num_modes, purpose_dim, trip_max_num, cond_num_dim, cond_cat_dims, Dz=BEST_DZ):
    return M.CVAE_StructuralZeros(
        num_modes=num_modes,
        purpose_dim=purpose_dim,
        trip_max_num=trip_max_num,
        cond_num_dim=cond_num_dim,
        cond_cat_dims=cond_cat_dims,
        hidden_layers=BEST_HIDDEN,
        latent_dim=Dz,
        length_distribution='lognormal'
    )

def train_model(model, train_loader, val_loader, lr=BEST_LR, wd=BEST_WD, beta=BEST_BETA):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_losses, val_losses, *_ = M.train_vae(
        model, train_loader, val_loader, optimizer,
        distribution_type='lognormal',
        epochs=100, beta=beta, patience=4,
        device='cpu', debug=False, annealation=True
    )
    return val_losses[-1]

def get_loaders(dataset, df, val_months, batch_size=128):
    val_mask  = df['DiaryMonth_raw'].isin(val_months)
    val_idx   = df.index[val_mask].tolist()
    train_idx = df.index[~val_mask].tolist()
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    # --- Load and prepare data ---
    df, _, _, catcondf, numcondf = load_data()

    # Preprocessing (same as train.py)
    for col in [f'trips_mode{i}' for i in range(1, 8)]:
        df = df[df[col] <= 10]
    df['DiaryMonth_raw'] = df['DiaryMonth'].copy()
    df['DiaryMonth'] = np.sin(2 * np.pi * (df['DiaryMonth'] / 12 + 0.25))
    for i in range(1, 8):
        df = df[~((df[f'trips_mode{i}'] > 0) & (df[f'mean_len_mode{i}'] == 0))]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare full dataset (default split just to get dataset object and scalers)
    xtp, xl, xpurp, x_num, x_cat, weights, _, _, scaler_len, scaler_cond, purpose_mapping, cat_cond_mapping = prepare_cvae_data_emb(
        df,
        trip_col=[f'trips_mode{i}' for i in range(1, 8)],
        length_col=[f'mean_len_mode{i}' for i in range(1, 8)],
        purpose_col='DayPrimTargetPurpV2',
        cat_cond_features=catcondf,
        num_cond_features=numcondf,
        batch_size=128
    )

    # Rebuild dataset object for custom splits
    dataset = TensorDataset(xtp, xl, xpurp, x_num, x_cat, weights)

    # Model dimensions
    num_modes      = xtp.shape[1]
    purpose_dim    = len(purpose_mapping)
    trip_max_num   = int(torch.max(xtp).item()) + 1
    cond_num_dim   = x_num.shape[1]
    cond_cat_dims  = [len(cat_cond_mapping[i]) for i in range(len(cat_cond_mapping))]

    results = []

    # ----------------------------------------------------------------
    # 1. SEED ROBUSTNESS — 5 seeds, fixed split [2, 6, 10]
    # ----------------------------------------------------------------
    print("\n=== SEED ROBUSTNESS ===")
    default_months = [2, 6, 10]
    train_loader, val_loader = get_loaders(dataset, df, default_months)

    for seed in [42, 123, 456]:
        set_seed(seed)
        model   = build_model(num_modes, purpose_dim, trip_max_num, cond_num_dim, cond_cat_dims)
        val_loss = train_model(model, train_loader, val_loader)
        tag      = f"seed{seed}"
        print(f"  [{tag}] Final val loss: {val_loss:.4f}")
        torch.save(model.state_dict(), path + f"models/robustness/cvae_{tag}.pth")
        results.append({'run': tag, 'val_loss': val_loss})

    # ----------------------------------------------------------------
    # 2. TEMPORAL SPLIT ROBUSTNESS — 3 alternative splits, seed=42
    # ----------------------------------------------------------------
    print("\n=== TEMPORAL SPLIT ROBUSTNESS ===")
    for val_months in [[1, 5, 9], [3, 7, 11], [4, 8, 12]]:
        set_seed(42)
        train_loader_tmp, val_loader_tmp = get_loaders(dataset, df, val_months)
        model    = build_model(num_modes, purpose_dim, trip_max_num, cond_num_dim, cond_cat_dims)
        val_loss = train_model(model, train_loader_tmp, val_loader_tmp)
        tag      = "split" + "_".join(map(str, val_months))
        print(f"  [{tag}] Final val loss: {val_loss:.4f}")
        torch.save(model.state_dict(), path + f"models/robustness/cvae_{tag}.pth")
        results.append({'run': tag, 'val_loss': val_loss})

    # # ----------------------------------------------------------------
    # 3. HYPERPARAMETER SENSITIVITY — Dz and beta, seed=42
    # ----------------------------------------------------------------
    print("\n=== HYPERPARAMETER SENSITIVITY ===")
    train_loader, val_loader = get_loaders(dataset, df, default_months)

    # for cfg in [{'Dz': 5, 'beta': 1, 'tag': 'Dz5'}, {'Dz': 20, 'beta': 1, 'tag': 'Dz20'}]:
    for cfg in [{'Dz': 20, 'beta': 1, 'tag': 'Dz20'}]:
        set_seed(42)
        model    = build_model(num_modes, purpose_dim, trip_max_num, cond_num_dim, cond_cat_dims, Dz=cfg['Dz'])
        val_loss = train_model(model, train_loader, val_loader, beta=cfg['beta'])
        tag      = cfg['tag']
        print(f"  [{tag}] Final val loss: {val_loss:.4f}")
        torch.save(model.state_dict(), path + f"models/robustness/cvae_{tag}.pth")
        results.append({'run': tag, 'val_loss': val_loss})

    # ----------------------------------------------------------------
    # Save results summary and scalers
    # ----------------------------------------------------------------
    pd.DataFrame(results).to_csv(path + "models/robustness/robustness_results2.csv", index=False)
    torch.save(scaler_cond, path + "models/robustness/scaler_cond.pth")
    torch.save(scaler_len,  path + "models/robustness/scaler_len.pth")
    print("\nDone. Results saved to models/robustness/")

if __name__ == "__main__":
    main()