# Synthetic Pseudo-Panel E-bike Diffusion

This repository contains the research code for generating and analysing
synthetic travel pseudo-panels for e-bike adoption research. The generative
model is a conditional variational autoencoder (CVAE) that models trips, trip lengths, purposes for a day. The repository also contains model training, generation, robustness
experiments, validation notebooks, and an LMDI decomposition implementation.

This is a **code-only paper-support release**. Raw survey data, geographic
data, generated datasets, trained weights, and local result folders are
excluded. The scripts document how the reported analyses were produced when the
required data and saved artifacts are available to the authors.

The code is released under the [MIT License](LICENSE). The license applies to
the repository's original code and documentation; it does not grant access to
the restricted source data or any third-party materials.

## Repository layout

```text
notebooks/
  adoption_analysis.ipynb
  lmdi_contributions.ipynb
  reallocation_analysis.ipynb
  validation_metrics.ipynb
src/
  model.py              CVAE architecture, loss, and training helpers
  train.py              Hyperparameter search and model training
  generate.py           Synthetic-panel generation
  generate_insample.py  In-sample generation from saved dataset splits
  robustness.py         Seed, split, and latent-dimension robustness runs
  robustness_no_embeddings.py
                        Robustness run without conditional embeddings
  lmdi.py               Transport decomposition calculations
```

## Requirements

The code is Python-based and uses:

- Python 3.11 or newer
- PyTorch
- NumPy and pandas
- scikit-learn and Optuna
- SciPy
- Matplotlib and seaborn
- GeoPandas and openpyxl
- Dask and Jupyter

## Typical workflow

The model-training stage expects a prepared model-ready dataset and feature
lists supplied through the authors' restricted data workflow. Once those
inputs are available, train the main model:

```powershell
python src\train.py
```

Generate synthetic panels after the required model and serialized dataset
artifacts have been restored:

```powershell
python src\generate.py
python src\generate_insample.py --help
```

Run robustness experiments with:

```powershell
python src\robustness.py
python src\robustness_no_embeddings.py
```

The notebooks cover adoption, reallocation, LMDI contributions, temporal
validation, and trip-length validation.


## Reproducibility status

This repository is an analysis and methods supplement, not a fully
self-contained reproduction package:

1. Raw input data, geographic data, and preprocessing code are external and
   excluded.
2. The training stage requires prepared model inputs supplied separately.
3. Dependency versions are not pinned.
4. There is no automated test suite or continuous-integration configuration.
5. The notebooks rely on generated files and local execution state.
