# Cross-Site Validation (Leave-One-Site-Out)
### Created by: Sylvia Imanirakiza

Please note this branch is a continuation of the site-specific analysis in the main branch, focusing on zero shot evaluation. 

This module checks how well the generator-usage models generalize to a healthcare facility they were not trained on. It uses leave-one-site-out (LOSO) cross-validation: the models are trained on all sites but one and tested on the site that was left out, and this is repeated for every site.

The sites included are:
1. Ndosho
2. Walikale
3. Kitatumba
4. Katwa

The models compared are:
1. Logistic Regression
2. Random Forest (default or tuned)
3. XG-Boost (default or tuned)

For every held-out site, model, and feature group, the script records accuracy, precision, recall, F1 score, the top-5 features, and the percentage error in detected generator on-time. The results are saved to:
*Modeling/summer25_results/cross_site_models/results_crossval.csv*

---

## Prepare Environment

To prepare the environment, follow these steps:

### 1. Create a Virtual Environment
```bash
cd Modeling
python -m venv virtualenv
source virtualenv/bin/activate # for mac
virtualenv\Scripts\activate # for windows
```

### 2. Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib xgboost seaborn openpyxl statsmodels hmmlearn
```

### 3. Verify installation
```bash
pip freeze
```

---

## Run the Cross-Site Validation

You can run the experiments in one of two ways.

### Option 1: Notebook
Open and run `Modeling/cross_site_validation.ipynb`.

### Option 2: Python script (recommended for quick testing)
Run these from the `Modeling` directory.

```bash
# Default Random Forest and XG-Boost (faster)
python cross_site_validation.py

# With hyperparameter tuning for Random Forest and XG-Boost (slower)
python cross_site_validation.py --tune_hyperparams
```

**Note:** `--root_path` should point to the folder that *contains* `Generator_ML_Emission_Modeling` (with a trailing `/`). The default is the original author's machine, so set it to your own path:
```bash
python cross_site_validation.py --root_path /path/to/GitHub/
```

---

## Plot the Results

```bash
# Plots for the default models
python cross_plots.py

# Plots for the tuned models
python cross_plots.py --tune_hyperparams
```

`cross_plots.py` reads `results_crossval.csv` and saves the generator on-time error and F1 score figures into the same `summer25_results/cross_site_models/` folder. It also takes `--root_path`, the same as above.

