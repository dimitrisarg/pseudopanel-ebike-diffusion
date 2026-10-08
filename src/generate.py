"""Generate synthetic travel panels from a trained conditional VAE.

The script loads model metadata, feature mappings, and scalers from a selected
experiment directory, then samples trip counts, purposes, and lengths
conditional on observed covariates. Data and trained model artifacts are
external inputs and are not included in this public code release.
"""
import torch
import numpy as np
import pandas as pd
import random
import ast
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)

import model as M
import os

dpath = Path(__file__).parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mode-specific maximum lengths based on observed data
MODE_MAX_LENGTHS = {
    0: 43.0,   # walk
    1: 128.0,  # bike
    2: 77.0,   # ebike
    3: 518.0,  # car
    4: 476.0,  # car_passenger
    5: 551.0,  # public transport
    6: 54.0,   # moped
}

NUM_SAMPLES = 100  # number of synthetic draws per observation
pop_size = 4
CVAE_name = "cvae_noembed.pth"
folder = "robustness_no_embed"
print(CVAE_name)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def set_model(model_dir, model_name, device='cpu'):
    """Load model parameters and best hyperparameters from files, then load the trained model."""
    with open(model_dir / "model_parameters.txt", "r") as f:
        num_modes    = int(f.readline().strip().split(": ")[1])
        purpose_dim  = int(f.readline().strip().split(": ")[1])
        trip_max_num = int(f.readline().strip().split(": ")[1])
        cond_num_dim = int(f.readline().strip().split(": ")[1])
        cond_cat_dims = eval(f.readline().strip().split(": ")[1])

    with open(model_dir / "best_hyperparameters.txt", "r") as f:
        param_str = f.read().splitlines()[0].split(":", 1)[1].strip()
    params = ast.literal_eval(param_str)
    Nl, Nn, Dz, _, _, _ = list(params.values())

    if model_name == 'cvae_Dz20.pth':
        Dz = 20
        print("Using latent dimension Dz=20 for the CVAE model.")
    elif model_name == 'cvae_Dz5.pth':
        Dz = 5
        print("Using latent dimension Dz=5 for the CVAE model.")
    
    use_embeddings = model_name != 'cvae_noembed.pth'
    if not use_embeddings:
        print("Loading no-embedding CVAE variant (one-hot conditionals/purpose).")

    hidden_layers = [Nn // (2**l) for l in range(Nl)]

    loaded_model = M.CVAE_StructuralZeros(
        num_modes=num_modes,
        purpose_dim=purpose_dim,
        trip_max_num=trip_max_num,
        cond_num_dim=cond_num_dim,
        cond_cat_dims=cond_cat_dims,
        hidden_layers=hidden_layers,
        latent_dim=Dz,
        length_distribution='lognormal',
        use_cond_embedding=use_embeddings,
        use_purpose_embedding=use_embeddings,
    ).to(device)

    loaded_model.load_state_dict(torch.load(model_dir / model_name, map_location=device))
    loaded_model.eval()
    return loaded_model

def decode_sample(loaded_model, z, x_num, x_cat):
    """Decode latent z into trips, lengths, purposes."""
    trips_logits, ln_mus, ln_sigmas, purpose_logits = loaded_model.decode(z, x_num, x_cat)

    # --- Trips (vectorized) ---
    # Stack logits: (batch, num_modes, num_trip_values)
    trip_probs = torch.stack(
        [torch.softmax(logits, dim=1) for logits in trips_logits], dim=1
    ).cpu().numpy()  # (batch, num_modes, num_trip_values)

    batch_size, num_modes, _ = trip_probs.shape
    # Sample all trips at once using cumsum + uniform draw
    u = np.random.uniform(size=(batch_size, num_modes, 1))
    trips_samples = (u > np.cumsum(trip_probs, axis=2)).sum(axis=2)  # (batch, num_modes)

    # --- Purpose (vectorized) ---
    purpose_probs = torch.softmax(purpose_logits, dim=1).cpu().numpy()
    u_p = np.random.uniform(size=(batch_size, 1))
    purposes = (u_p > np.cumsum(purpose_probs, axis=1)).sum(axis=1)

    # --- Lengths (vectorized with rejection sampling) ---
    ln_mus_tensor    = torch.cat(ln_mus,    dim=1)  # (batch, num_modes)
    ln_sigmas_tensor = torch.cat(ln_sigmas, dim=1)  # (batch, num_modes)

    # Build max_len array: (1, num_modes) broadcast over batch
    max_lens = torch.tensor(
        [MODE_MAX_LENGTHS.get(m, 100.0) for m in range(num_modes)],
        dtype=torch.float32
    ).unsqueeze(0)  # (1, num_modes)

    # Active mask: only sample where trips > 0
    active_mask = torch.tensor(trips_samples > 0, dtype=torch.bool)  # (batch, num_modes)

    # Vectorized rejection sampling — draw max_attempts candidates at once
    max_attempts = 100
    dist = torch.distributions.LogNormal(ln_mus_tensor, ln_sigmas_tensor)
    
    # Draw all candidates: (max_attempts, batch, num_modes)
    candidates = dist.sample((max_attempts,))
    
    # Find first valid draw per (batch, mode): valid = below max_len
    valid = candidates < max_lens.unsqueeze(0)  # (max_attempts, batch, num_modes)
    
    # First valid index per cell
    has_valid = valid.any(dim=0)  # (batch, num_modes)
    first_valid_idx = valid.long().argmax(dim=0)  # (max_attempts, batch, num_modes) -> (batch, num_modes)
    
    # Gather first valid sample
    lengths_tensor = candidates.gather(
        0, first_valid_idx.unsqueeze(0)
    ).squeeze(0)  # (batch, num_modes)
    
    # Where no valid draw found, clamp to 95% of max
    clamp_mask = active_mask & ~has_valid
    clamp_vals = max_lens.expand(batch_size, -1) * 0.95
    lengths_tensor = torch.where(clamp_mask, clamp_vals, lengths_tensor)

    # Warn if clamping rate is high
    clamp_counts  = clamp_mask.sum(dim=0)   # (num_modes,)
    active_counts = active_mask.sum(dim=0)  # (num_modes,)
    mode_names = ['walk', 'bike', 'ebike', 'car', 'car_passenger', 'pt', 'moped']
    for m in range(num_modes):
        if active_counts[m] > 0:
            rate = clamp_counts[m].item() / active_counts[m].item()
            if rate > 0.01:
                warnings.warn(
                    f"[decode_sample] Mode '{mode_names[m]}': "
                    f"{clamp_counts[m].item()}/{active_counts[m].item()} lengths clamped "
                    f"({rate:.1%}). Distribution may be miscalibrated.",
                    RuntimeWarning, stacklevel=2
                )

    # Apply structural zeros
    lengths_tensor = lengths_tensor * active_mask.float()

    return trips_samples, lengths_tensor.cpu().numpy(), purposes

def sample_SPP(x_num, x_cat, unid, model_dir, num_samples=100, start_sample=0, output_dir=None, device='cpu'):
    """Generate SPP samples using individual-level latent z."""
    loaded_model = set_model(model_dir, CVAE_name, device)
    output_dir = output_dir or dpath / f"results/samples_{pop_size}_{CVAE_name.split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        unique_ids, inverse_indices = torch.unique(unid, return_inverse=True)

        print(f"Generating {num_samples} SPP samples for {len(unique_ids)} unique individuals...")
        for s in range(start_sample, num_samples):
            set_seed(s)

            z_unique = torch.randn(len(unique_ids), loaded_model.latent_dim)
            z_full   = z_unique[inverse_indices]

            trips, lengths, purposes = decode_sample(loaded_model, z_full, x_num, x_cat)

            num_modes = trips.shape[1]
            df = pd.DataFrame({
                **{f"trips_mode_{i+1}":   trips[:, i]   for i in range(num_modes)},
                **{f"lengths_mode_{i+1}": lengths[:, i] for i in range(num_modes)},
                "purposes": purposes
            })

            file_path = output_dir / f"spp_samples_{s+1}.parquet"
            df.to_parquet(file_path, index=False)

            if (s+1) % 10 == 0 or s == 0 or s == num_samples - 1:
                print(f"  Generating sample progress {(s+1)/ num_samples * 100} % ...")

def load_data():
    """Load panel data and feature column names."""
    df = pd.read_parquet(dpath / f'data/TUmodeldata_panel_ally_{pop_size}.parquet', engine='auto')

    with open(dpath / "data/pref_cat.txt")  as f: cat_features     = f.read().splitlines()
    with open(dpath / "data/pref_num.txt")  as f: num_features     = f.read().splitlines()
    with open(dpath / "data/con_num.txt")   as f: num_cond_features = f.read().splitlines()
    with open(dpath / "data/con_cat.txt")   as f: cat_cond_features = f.read().splitlines()

    id_index = df['NewSessionId'].copy()
    df = df[cat_features + num_features + cat_cond_features + num_cond_features].copy()

    return df, cat_features, num_features, cat_cond_features, num_cond_features, id_index


if __name__ == "__main__":
    set_seed(42)

    df, cat_features, num_features, cat_cond_features, num_cond_features, id_index = load_data()
    individual_ids = torch.tensor(id_index.values)
    print(f"Loaded panel data: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Fix the month scale before transforming
    df['DiaryMonth'] = np.sin(2 * np.pi * (df['DiaryMonth'] / 12 + 0.25))

    # Apply categorical mapping
    cat_cond_mapping = torch.load(dpath / 'results/datasets/cat_cond_mapping.pth', weights_only=False)
    for i, col in enumerate(cat_cond_features):
        df[col] = df[col].map(cat_cond_mapping[i])

    # Tensors
    cat_cond_tensor = torch.tensor(df[cat_cond_features].values, dtype=torch.long,    device=device)
    scaler_cond     = torch.load(dpath / "models"/  folder / "scaler_cond.pth", weights_only=False)
    num_cond_tensor = torch.tensor(scaler_cond.transform(df[num_cond_features]), dtype=torch.float32, device=device)

    # Generate
    sample_SPP(
        x_num=num_cond_tensor,
        x_cat=cat_cond_tensor,
        unid=individual_ids,
        model_dir= dpath / "models" / folder,
        num_samples=NUM_SAMPLES,
        start_sample=0,
        device=device
    )