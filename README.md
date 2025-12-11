# Physics-constrained hyperspectral soil moisture modeling in red clay (20250104Study)

This repository contains the core data and source code for the experiment
conducted on **January 4, 2025**, supporting the manuscript:

> Xing, M., et al. **"Improving soil moisture prediction through physics-constrained hyperspectral modeling in red clay environments"**, submitted to *Computers & Geosciences*.

The workflow combines:

- Physical soil dielectric constant (SDC) models;
- Hyperspectral feature selection and machine-learning models;
- Pure data-driven models;
- Hybrid physics–data models for soil moisture prediction in red clay environments.

---

## Repository contents

The main files in this repository are:

- **`2.xlsx`**  
  Core dataset containing volumetric soil water content and UAV-based hyperspectral reflectance.
- **`2_SG平滑.xlsx`**  
  Output of spectral smoothing (Savitzky–Golay), used as input for subsequent modeling steps.

- **`0SG平滑.py`**  
  Applies Savitzky–Golay smoothing and basic preprocessing to the hyperspectral reflectance data.  
  Reads `2.xlsx` and can optionally export smoothed spectra to `2_SG平滑.xlsx`.

- **`1皮尔逊+Lasso.py`**  
  Performs feature selection using Pearson correlation and Lasso regression to identify
  water-sensitive spectral bands.

- **`1皮尔逊+Lasso_SDC.py`**  
  Extends feature selection by incorporating soil dielectric constant (SDC) information,
  preparing features for physics-constrained models.

- **`fit.py` / `fit2.py`**  
  Scripts for fitting dielectric-based physical soil moisture models
  (e.g., Topp-type or related SDC–SMC relationships) to the experimental data.

- **`RF_SVR_LightGBM.py`**  
  Trains and evaluates ensemble machine-learning models (Random Forest, SVR, LightGBM, etc.)
  based on the selected hyperspectral features.  
  This script is used as the **quick-test** entry point (see section “Quick test” below).

- **`RF_SVR_lightGBM.py`**  
  Alternative implementation and/or extended version of the RF–SVR–LightGBM workflow,
  including grid search or additional parameter settings.

- **`反演.py`**  
  Implements the hybrid physics-constrained inversion framework, combining dielectric
  model outputs with hyperspectral ML predictions.

- **`反演图.py`**  
  Generates key figures for the manuscript (e.g., prediction–observation scatter plots,
  spatial patterns, or comparison of different model types).

- **`评价.py`**  
  Computes and summarizes model performance metrics (e.g., R², RMSE, RPD) and writes them
  to the Excel summary files.

- **`GRADM.xlsx`**, **`SMC_models_with_gridsearch.xlsx`**, **`模型对比.xlsx`**, **`Metrics.xlsx`**  
  Excel workbooks containing model configurations, grid-search results, and performance
  comparison among physical, data-driven, and hybrid models.

- **`Data and code.zip`**  
  Legacy archive of the same data and scripts, kept only for completeness.
  Users are encouraged to use the unpacked files listed above instead of the zip file.

---

## Environment

The codes were developed and tested with:

- **Python 3.11**

A minimal set of required Python packages includes:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `xgboost`
- `lightgbm`

Additional packages (e.g., `torch` / `tensorflow`) may be needed if deep-learning
models are activated in some scripts.


## Quick test

To verify that the code works and to reproduce a minimal version of the workflow:

```bash
git clone https://github.com/xingmingjie88/20250104Study.git
cd 20250104Study

# Activate the environment first
conda activate smc_redclay     # or your own environment name

# Run the quick test
python quick_test/run_quick_test.py


Example installation using `pip`:

```bash
pip install numpy pandas scikit-learn scipy matplotlib xgboost lightgbm
