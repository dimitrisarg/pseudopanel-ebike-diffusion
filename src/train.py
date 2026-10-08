"""Train the conditional VAE from prepared model inputs.

The restricted data workflow must first provide the model-ready table and
feature-list files under ``data/``. This module prepares tensors, performs the
Optuna hyperparameter search, trains the selected model, and writes model
metadata and preprocessing objects to the local experiment directories.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
import os
import sys
import model as M
import optuna
import random
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = sys.stdout

torch.manual_seed(42)  # reproducibility
np.random.seed(42)  # reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
g = torch.Generator()
g.manual_seed(42)

path = '../'

def load_data():

    # Load TU dataset
    df = pd.read_csv(path + "data\TUmodeldata_full.csv")

    # Load column names
    with open(path +"data/pref_cat.txt", "r") as f:
        cat_features = f.read().splitlines()
    with open(path +"data/pref_num.txt", "r") as f:
        num_features = f.read().splitlines()
    with open(path +"data/con_num.txt", "r") as f:
        num_cond_features = f.read().splitlines()
    with open(path +"data/con_cat.txt", "r") as f:
        cat_cond_features = f.read().splitlines()

    # Reorder columns
    df = df[cat_features + num_features + cat_cond_features + num_cond_features + ['SessionWeight']].copy()

    return df, cat_features, num_features, cat_cond_features, num_cond_features

def prepare_cvae_data_emb(df, trip_col, length_col, purpose_col, cat_cond_features, num_cond_features, batch_size=128, train_ratio=0.75, seed=42, save_path=None):
    """
    Prepare data for CVAE using embeddings:
    - Trips (categorical, integer indices)
    - Trip lengths (continuous)
    - Purpose (categorical, integer indices)
    - Conditional numerical and categorical features (numerical normalized, categorical as indices)
    Returns tensors and DataLoaders.
    """
 
    # --- Trips ---
    x_trips = torch.tensor(df[trip_col].astype(int).values, dtype=torch.long)

    # --- Trip lengths ---
    scaler_len = StandardScaler()
    x_lengths = torch.tensor(df[length_col].values, dtype=torch.float32)

    # --- Purpose ---
    purpose_mapping = {cat: i for i, cat in enumerate(df[purpose_col].unique())}
    x_purpose = torch.tensor(df[purpose_col].map(purpose_mapping).values, dtype=torch.long)

    # --- Conditional numerical features ---
    scaler_cond = StandardScaler()
    x_num = scaler_cond.fit_transform(df[num_cond_features])
    x_num = torch.tensor(x_num, dtype=torch.float32)

    # --- Conditional categorical features ---
    cat_cond_mapping = []
    x_cat_indices = []
    for col in cat_cond_features:
        mapping = {cat: i for i, cat in enumerate(df[col].unique())}
        cat_cond_mapping.append(mapping)
        x_cat_indices.append(torch.tensor(df[col].map(mapping).values, dtype=torch.long))
    
    # Combine categorical indices along last dimension
    x_cat = torch.stack(x_cat_indices, dim=1)  # shape: (n_samples, n_cat_cond)
    
    # --- Sample weights ---
    #sample_weights = torch.tensor(df['SessionWeight'].values, dtype=torch.float32)
    # Use a tensor of ones instead of actual weights
    sample_weights = torch.ones(len(df), dtype=torch.float32)

    # --- Create dataset ---
    dataset = TensorDataset(x_trips, x_lengths, x_purpose, x_num, x_cat, sample_weights)

    # --- Temporal split using raw month (before sine transform) ---
    # val_months = [2, 5, 8, 11] # February, May, August, November
    val_months = [2, 6, 10] # February, June, October
    val_mask  = df['DiaryMonth_raw'].isin(val_months)
    val_idx   = df.index[val_mask].tolist()
    train_idx = df.index[~val_mask].tolist()

    print(f"Temporal split — Val months: {val_months}")
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    print(f"Train ratio: {len(train_idx)/len(df):.2f}, Val ratio: {len(val_idx)/len(df):.2f}")
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset   = Subset(dataset, val_idx)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if save_path is not None:
        # Save dataset, x_num, x_cat, train, xtrips, xlengths, xpurpose to files
        torch.save(x_num, os.path.join(save_path + 'data/inout', "x_num.pth"))
        torch.save(x_cat, os.path.join(save_path + 'data/inout', "x_cat.pth"))
        torch.save(x_trips, os.path.join(save_path + 'data/inout', "x_trips.pth"))
        torch.save(x_lengths, os.path.join(save_path + 'data/inout', "x_lengths.pth"))
        torch.save(x_purpose, os.path.join(save_path + 'data/inout', "x_purpose.pth"))
        torch.save(sample_weights, os.path.join(save_path + 'data/inout', "sample_weights.pth"))
        torch.save(dataset, os.path.join(save_path + 'data/inout', "dataset.pth"))
        # Save mapper dictionaries
        torch.save(purpose_mapping, os.path.join(save_path + 'data/inout', "purpose_mapping.pth"))
        torch.save(cat_cond_mapping, os.path.join(save_path + 'data/inout', "cat_cond_mapping.pth"))
        # Save train and validation datasets
        torch.save(train_dataset, os.path.join(save_path + 'data/inout', "train_dataset.pth"))
        torch.save(val_dataset, os.path.join(save_path + 'data/inout', "val_dataset.pth"))

    return x_trips, x_lengths, x_purpose, x_num, x_cat, sample_weights, train_loader, val_loader, scaler_len, scaler_cond, purpose_mapping, cat_cond_mapping


def main() -> None:

    df, catf, numf, catcondf, numcondf = load_data()


    # Treat month as cyclical variable
    df['DiaryMonth_raw'] = df['DiaryMonth'].copy()  # raw integer
    df['DiaryMonth'] = np.sin(2 * np.pi * (df['DiaryMonth'] / 12 + 0.25))

    # Remove rows where trips > 0 but mean length = 0
    for i in range(1, 8):
        trips_col = f'trips_mode{i}'
        len_col = f'mean_len_mode{i}'
        df = df[~((df[trips_col] > 0) & (df[len_col] == 0))]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(path + 'data/TUmodeldata_full_cleaned.csv', index=False)

    xtp, xl, xpurp, x_num, x_cat, weights ,train_loader, val_loader, scaler_len, scaler_cond, purpose_mapping, cat_cond_mapping = prepare_cvae_data_emb(
        df, 
        trip_col=['trips_mode1', 'trips_mode2', 'trips_mode3', 'trips_mode4', 'trips_mode5', 'trips_mode6', 'trips_mode7'], 
        length_col=['mean_len_mode1', 'mean_len_mode2', 'mean_len_mode3', 'mean_len_mode4', 'mean_len_mode5', 'mean_len_mode6', 'mean_len_mode7'], 
        purpose_col='DayPrimTargetPurpV2', 
        cat_cond_features=catcondf, 
        num_cond_features=numcondf, 
        batch_size=128,
        save_path=path
    )

    # Parameters
    num_modes = xtp.shape[1] # number of trip modes 
    purpose_dim = len(purpose_mapping) # number of purpose categories 
    trip_max_num = int(torch.max(xtp).item()) + 1 # max trips across all modes 
    cond_num_dim = x_num.shape[1] if x_num is not None else 0 
    cond_cat_dims = [len(cat_cond_mapping[i]) for i in range(len(cat_cond_mapping))]

    # Hyperparameter Grid Search
    def objective(trial):
        # Suggest hyperparameters
        Nl = trial.suggest_categorical("Nl", [1, 2, 3])
        Nn = trial.suggest_categorical("Nn", [64, 128, 256])
        Dz = trial.suggest_categorical("Dz", [5, 10, 15, 25])
        beta = trial.suggest_categorical("beta", [0.1, 0.5, 1, 10])
        lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 0])
        epochs = 100  # or early stopping inside
        hidden_layers = [Nn // (2**l) for l in range(Nl)]

        
        # Instantiate model
        model = M.CVAE_StructuralZeros(
            num_modes=num_modes,
            purpose_dim=purpose_dim,
            trip_max_num=trip_max_num,
            cond_num_dim=cond_num_dim,
            cond_cat_dims=cond_cat_dims,
            hidden_layers=hidden_layers,
            latent_dim=Dz,
            length_distribution= 'lognormal'  # 'lognormal' or 'gamma' or 'beta'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train model with optional early stopping
        train_losses, val_losses, *_ = M.train_vae(
            model, train_loader, val_loader, optimizer,
            distribution_type = 'lognormal', 
            epochs=epochs, beta=beta, patience=4, 
            device='cpu', debug = False, annealation=True
        )

        final_val_loss = val_losses[-1]
        trial.set_user_attr("model", model)
        return final_val_loss
    

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Retrieve best model and hyperparameters
    best_trial = study.best_trial
    best_model = best_trial.user_attrs["model"]
    best_params = best_trial.params

    print("Best Hyperparameters:", best_params)
    # Save best model
    torch.save(best_model.state_dict(), path + "models/inout/best_cvae_model.pth")
    # Save the hyperparameters
    with open(path + "models/inout/best_hyperparameters.txt", "w") as f:
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Final Loss: {best_trial}\n")

    # Save the scaler and encoders
    torch.save(scaler_cond, path + "models/inout/scaler_cond.pth")
    torch.save(scaler_len, path + "models/inout/scaler_len.pth")

    # Save to text file
    with open( path + "models/inout/model_parameters.txt", "w") as f:
        f.write(f"num_modes: {num_modes}\n")
        f.write(f"purpose_dim: {purpose_dim}\n")
        f.write(f"trip_max_num: {trip_max_num}\n")
        f.write(f"cond_num_dim: {cond_num_dim}\n")
        f.write(f"cond_cat_dims: {cond_cat_dims}\n")

if __name__ == "__main__":
    main()
