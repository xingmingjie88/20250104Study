# 20250104Study
The data and paper code of the experiment conducted on January 4, 2025

This repository contains the core code and data for a research project on soil moisture estimation using UAV-based hyperspectral imagery and dielectric-constant-constrained hybrid models. The workflow combines:

- Physical soil dielectric constant (SDC) models
- Machine-learning models based on hyperspectral features
- Pure data-driven models
- Hybrid physical–data models

All raw soil moisture and hyperspectral data are stored in **`2.xlsx`** (e.g., volumetric soil water content and corresponding reflectance bands). Please adapt the column names in the scripts to your own file if needed.

---

## Directory Structure & Workflow

**Root directory**

- **`2.xlsx`**  
  Raw dataset containing soil volumetric water content and hyperspectral reflectance.

- **`0SG平滑.py`**  
  Savitzky–Golay smoothing and basic preprocessing of spectral curves.  
  Outputs smoothed spectra for later steps.

- **`1皮尔逊+Lasso.py`**  
  Feature selection based on Pearson correlation and Lasso regression to identify sensitive hyperspectral bands.

- **`1皮尔逊+Lasso_SDC.py`**  
  Feature selection with additional soil dielectric constant (SDC) constraints, preparing inputs for hybrid physical–data models.

**Model folders**

1. **`1SDC_Model/`**  
   Implementation and fitting of soil dielectric constant models (e.g., Topp, Herkelrath, CRIM).  
   Uses observed soil moisture to calibrate dielectric models and evaluate their accuracy and stability.

2. **`2SHR_ML/`**  
   Hyperspectral machine-learning models.  
   Trains RF, SVR, BPNN, LightGBM, etc. using selected spectral features to estimate soil moisture.

3. **`3Data_Model/`**  
   Pure data-driven models.  
   Uses multi-source inputs (spectral features, SDC estimates, etc.) to build more complex regression models and compare different feature sets/algorithms.

4. **`4Hybrid_model/`**  
   Hybrid physical–data models.  
   Combines outputs of dielectric constant models with machine-learning models (e.g., using SDC as extra features or residual-correction terms) to improve generalization and physical consistency.

**Recommended execution order**

```text
0SG平滑.py
→ 1皮尔逊+Lasso.py and/or 1皮尔逊+Lasso_SDC.py
→ 1SDC_Model/
→ 2SHR_ML/
→ 3Data_Model/
→ 4Hybrid_model/
