"""Train the no-embedding CVAE comparison model.

This is a controlled robustness run using one-hot categorical and purpose
inputs instead of learned embeddings, with the same prepared data and default
validation split as ``robustness.py``.
"""
import torch
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
os.makedirs(path + "models/robustness_no_embed/", exist_ok=True)

# --- Best hyperparameters from Trial 42 (same as embedding version) ---
BEST_LR      = 0.0003
BEST_WD      = 0.0001
BEST_DZ      = 15
BEST_BETA    = 1
BEST_HIDDEN  = [256]  # Nl=2, Nn=64
SEED         = 42
VAL_MONTHS   = [2, 6, 10]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(dataset, df, val_months, batch_size=128):
    val_mask  = df['DiaryMonth_raw'].isin(val_months)
    val_idx   = df.index[val_mask].tolist()
    train_idx = df.index[~val_mask].tolist()
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    # --- Load and prepare data (identical preprocessing to robustness.py) ---
    df, _, _, catcondf, numcondf = load_data()

    for col in [f'trips_mode{i}' for i in range(1, 8)]:
        df = df[df[col] <= 10]
    df['DiaryMonth_raw'] = df['DiaryMonth'].copy()
    df['DiaryMonth'] = np.sin(2 * np.pi * (df['DiaryMonth'] / 12 + 0.25))
    for i in range(1, 8):
        df = df[~((df[f'trips_mode{i}'] > 0) & (df[f'mean_len_mode{i}'] == 0))]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Note: prepare_cvae_data_emb still returns integer-coded x_cat / x_purpose.
    # That's fine — with use_cond_embedding=False / use_purpose_embedding=False
    # the model one-hot encodes these internally (see embed_conditionals /
    # process_purpose in model.py), so no changes are needed here.
    xtp, xl, xpurp, x_num, x_cat, weights, _, _, scaler_len, scaler_cond, purpose_mapping, cat_cond_mapping = prepare_cvae_data_emb(
        df,
        trip_col=[f'trips_mode{i}' for i in range(1, 8)],
        length_col=[f'mean_len_mode{i}' for i in range(1, 8)],
        purpose_col='DayPrimTargetPurpV2',
        cat_cond_features=catcondf,
        num_cond_features=numcondf,
        batch_size=128
    )

    dataset = TensorDataset(xtp, xl, xpurp, x_num, x_cat, weights)

    num_modes      = xtp.shape[1]
    purpose_dim    = len(purpose_mapping)
    trip_max_num   = int(torch.max(xtp).item()) + 1
    cond_num_dim   = x_num.shape[1]
    cond_cat_dims  = [len(cat_cond_mapping[i]) for i in range(len(cat_cond_mapping))]

    train_loader, val_loader = get_loaders(dataset, df, VAL_MONTHS)

    # --- Build and train model without embeddings ---
    set_seed(SEED)
    model = M.CVAE_StructuralZeros(
        num_modes=num_modes,
        purpose_dim=purpose_dim,
        trip_max_num=trip_max_num,
        cond_num_dim=cond_num_dim,
        cond_cat_dims=cond_cat_dims,
        hidden_layers=BEST_HIDDEN,
        latent_dim=BEST_DZ,
        length_distribution='lognormal',
        use_cond_embedding=False,     # <-- one-hot categorical conditionals
        use_purpose_embedding=False,  # <-- one-hot purpose
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=BEST_WD)
    train_losses, val_losses, *_ = M.train_vae(
        model, train_loader, val_loader, optimizer,
        distribution_type='lognormal',
        epochs=100, beta=BEST_BETA, patience=4,
        device='cpu', debug=False, annealation=True
    )

    print(f"\nFinal val loss (no embeddings): {val_losses[-1]:.4f}")

    torch.save(model.state_dict(), path + "models/robustness_no_embed/cvae_noembed.pth")
    torch.save(scaler_cond, path + "models/robustness_no_embed/scaler_cond.pth")
    torch.save(scaler_len,  path + "models/robustness_no_embed/scaler_len.pth")
    print("Done. Model saved to models/robustness_no_embed/")

if __name__ == "__main__":
    main()