"""
Generate synthetic samples for the train and validation splits using
the observed conditioning variables (x_num, x_cat) from the saved
datasets. Output is saved as .npz files matching the format expected
by the validation notebook (Validation_metrics.ipynb):

    trips:    (num_samples, batch, num_modes)
    lengths:  (num_samples, batch, num_modes)
    purposes: (num_samples, batch)
    mileage:  (num_samples, batch, num_modes)

Usage:
    python generate_insample.py               # generates both splits
    python generate_insample.py --split train # train only
    python generate_insample.py --split val   # val only
"""
import argparse
import numpy as np
import os
from pathlib import Path
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from generate import set_seed, set_model, decode_sample

dpath  = Path(__file__).parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = 100  # number of synthetic draws per observation
CVAE_name = "best_cvae_model.pth"
folder = "lognorm"

# -----------------------------------------------------------------------
# Core generation function
# -----------------------------------------------------------------------

def generate_insample(x_num, x_cat, loaded_model, num_samples=NUM_SAMPLES, device='cpu'):
    """
    Generate synthetic samples conditioning on observed x_num, x_cat.

    Returns dict with:
        trips:    (num_samples, batch, num_modes)
        lengths:  (num_samples, batch, num_modes)
        purposes: (num_samples, batch)
        mileage:  (num_samples, batch, num_modes)
    """
    all_trips, all_lengths, all_purposes = [], [], []

    with torch.no_grad():
        for s in range(num_samples):
            set_seed(s)
            if (s+1) % 10 == 0 or s == 0 or s == num_samples - 1:
                print(f"  Generating sample progress {(s+1)/ num_samples * 100} % ...") 

            # Fresh z per draw — no individual-level z here,
            # since we are evaluating reconstruction, not panel generation
            z = torch.randn(x_cat.size(0), loaded_model.latent_dim).to(device)

            trips, lengths, purposes = decode_sample(loaded_model, z, x_num, x_cat)

            all_trips.append(trips)
            all_lengths.append(lengths)
            all_purposes.append(purposes)

    trips_arr    = np.stack(all_trips)           # (num_samples, batch, num_modes)
    lengths_arr  = np.stack(all_lengths)         # (num_samples, batch, num_modes)
    purposes_arr = np.stack(all_purposes)        # (num_samples, batch)
    mileage_arr  = trips_arr * lengths_arr       # (num_samples, batch, num_modes)

    return {
        'trips':    trips_arr,
        'lengths':  lengths_arr,
        'purposes': purposes_arr,
        'mileage':  mileage_arr
    }


def load_split(data_split):
    """Load x_num, x_cat from saved train or val dataset."""
    dataset_path = dpath / f"results/datasets/{data_split}_dataset.pth"
    dataset = torch.load(dataset_path, weights_only=False)
    x_trips, x_lengths, x_purpose, x_num, x_cat, w = dataset[:]
    print(f"  Loaded {data_split} split: {x_num.shape[0]} observations")
    return x_num.to(device), x_cat.to(device), x_trips, x_lengths, x_purpose


def save_npz(synthetic, output_path):
    """Save synthetic dict as compressed .npz."""
    os.makedirs(output_path.parent, exist_ok=True)
    np.savez_compressed(output_path, **synthetic)
    print(f"  Saved → {output_path}")

# ----------------------------------------------------------------------
# Single model — train + val
# -----------------------------------------------------------------------

def run_single(splits, model_dir, output_dir, num_samples=NUM_SAMPLES):
    """Generate insample data for the best model."""
    print(f"\n=== Loading model from {model_dir} ===")
    loaded_model = set_model(model_dir, CVAE_name, device)

    for split in splits:
        print(f"\n--- Split: {split} ---")
        x_num, x_cat, x_trips, x_lengths, x_purpose = load_split(split)

        synthetic = generate_insample(x_num, x_cat, loaded_model,
                                      num_samples=num_samples, device=device)

        # Also save ground truth alongside synthetic for convenience
        synthetic['gt_trips']   = x_trips.numpy()
        synthetic['gt_lengths'] = x_lengths.numpy()
        synthetic['gt_purpose'] = x_purpose.numpy()
        synthetic['gt_mileage'] = (x_trips * x_lengths).numpy()

        save_npz(synthetic, output_dir / f"synthetic_{split}.npz")

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both',
                        help="Which split to generate (default: both)")
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                        help=f"Number of synthetic draws (default: {NUM_SAMPLES})")
    args = parser.parse_args()

    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    set_seed(42)
    run_single(
        splits=splits,
        model_dir=dpath / "models" / folder,
        output_dir=dpath / f"results/val_samples_{CVAE_name.split('.')[0]}",
        num_samples=args.num_samples
    )

    print("\nDone.")