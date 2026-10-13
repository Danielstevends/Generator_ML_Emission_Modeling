from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import utils
import os
from importlib import reload
# from utils import g
reload(utils)

import argparse
# Argument parser for command line options
parser = argparse.ArgumentParser(description="Cross-site validation for generator data")
parser.add_argument('--tune_hyperparams', action='store_true', help="Enable hyperparameter tuning")
parser.add_argument('--root_path', type=str, default="/Users/sylviaimanirakiza/Documents/GitHub/", help="Root path of the project")
# Parse command line arguments
args = parser.parse_args()
# Check if hyperparameter tuning is enabled
tune_hyperparams = args.tune_hyperparams
#check if rootpath exists
root_path = args.root_path
if os.path.exists(root_path):
    print(f"Root path exists {root_path}")
else:
    print(f"Root path does not exist: {root_path}. Please check the path.")

print(f"Hyperparameter tuning is {'enabled' if tune_hyperparams else 'disabled'}.")


print("=================================================================================")
# Your existing constants and setup
minutes_interval = 2
NOx_lb_hp_hr = 0.024
CO_lb_hp_hr = 0.0055
SOx_lb_hp_hr = 0.00809
CO2_lb_hp_hr = 1.16
PM_lb_hp_hr = 0.0007
kwh_to_hp_hr = 1.341
lb_to_gr = 453.592

emissions = ['NOx', 'CO', 'SOx', 'CO2', 'PM']

# Your existing site setup (keeping your original paths and configuration)
sites = ["Ndosho", "Walikale", "Kitatumba", "Katwa"]

data_paths = [f"{root_path}Generator_ML_Emission_Modeling/data/0. ChCbNdosho/training",
            f"{root_path}Generator_ML_Emission_Modeling/data/1. Combination 1_Walikale/training",
              f"{root_path}Generator_ML_Emission_Modeling/data/5. Combination 5/training/kitatumba",
                f"{root_path}Generator_ML_Emission_Modeling/data/5. Combination 5/training/katwa"]

site_columns_oi = {
   "Walikale":[['Frequency 09B35150 (Hgr Walikale - Lab)',
                        'Frequency 64B734C2 (Hgr Walikale - Cabane)',
                        'Voltage 09B35150 (Hgr Walikale - Lab)',
                        'Voltage 64B734C2 (Hgr Walikale - Cabane)'
                        ], data_paths[1] ],
     "Ndosho": [[ 'Frequency 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                'Frequency C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)',
                'Voltage 3E2F4FD3 (Ch Cbca Bethesda Ndosho - laboratoire)',
                'Voltage C30F2E03 (Ch Cbca Bethesda Ndosho - clinique)'
                ],data_paths[0]],
      "Katwa":[['Frequency 1C712FB8 (Hgr Katwa - Labo)',
                'Frequency AEBA416D (Hgr Katwa - Cabine)',
                'Voltage 1C712FB8 (Hgr Katwa - Labo)',
                'Voltage AEBA416D (Hgr Katwa - Cabine)'
                ], data_paths[3]],
    "Kitatumba": [['Frequency 6EF39700 (Hgr Kitatumba - Labo)',
                   'Frequency D2B9CB2E (Hgr Kitatumba - Mdh office)',
                   'Voltage 6EF39700 (Hgr Kitatumba - Labo)',
                   'Voltage D2B9CB2E (Hgr Kitatumba - Mdh office)'
    ], data_paths[2]]
}

# Load and process data (keeping your original preprocessing)
site_names = sites 
site_dfs = {site: utils.load_data(path) for site, path in zip(site_names, data_paths)}

def rename_site_columns(df, site):
    freq_cols = site_columns_oi[site][0][:2]
    volt_cols = site_columns_oi[site][0][2:]
    rename_map = {
        freq_cols[0]: 'freq_a',
        freq_cols[1]: 'freq_b',
        volt_cols[0]: 'volt_a',
        volt_cols[1]: 'volt_b'
    }
    df = df.rename(columns=rename_map)
    df['Site'] = site
    return df

processed_dfs = []
for site, df in site_dfs.items():
    processed_df = rename_site_columns(df, site)
    processed_dfs.append(processed_df)

combined_df = pd.concat(processed_dfs, ignore_index=True)
combined_df['Time'] = pd.to_datetime(combined_df['Time'])

# Feature sets

feature_sets = {
    "Voltage, Frequency": ['freq_a', 'freq_b', 'volt_a', 'volt_b',],
    "Voltage, Frequency, Mean Threshold":['freq_a', 'freq_b', 'volt_a', 'volt_b',
                                             'higher_freq_lab', 'higher_freq_clinique', 'higher_volt_lab','higher_volt_clinique'
                                                ],
    "Voltage, Frequency, Mean Threshold, Time Series": ['freq_a', 'freq_b', 'volt_a', 'volt_b',
                                                        'higher_freq_lab', 'higher_freq_clinique', 'higher_volt_lab',
                                                        'higher_volt_clinique', 'Freq_delta_lab', 'Freq_delta_clinique',
                                                        'Volt_delta_lab', 'Volt_delta_clinique', 'freq_delta_negative_lab',
                                                        'freq_delta_negative_clinique', 'volt_delta_positive_lab',
                                                        'volt_delta_positive_clinique', 'freq_in_range_lab',
                                                        'volt_in_range_lab', 'freq_in_range_clinique', 'volt_in_range_clinique',
                                                        'Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
                                                        'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
                                                        'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18',
                                                        'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23',
                                                        'High_delta_freq_lab', 'High_delta_freq_clinique',
                                                        'High_delta_volt_lab', 'High_delta_volt_clinique',
                                                        'Prev_High_delta_freq_lab', 'Prev_High_delta_freq_clinique',
                                                        'Prev_High_delta_volt_lab', 'Prev_High_delta_volt_clinique'],
                                                                                     
  "Voltage, Frequency, Mean Threshold, Time Series, HMM":['freq_a', 'freq_b', 'volt_a', 'volt_b',
                                                        'higher_freq_lab', 'higher_freq_clinique', 'higher_volt_lab',
                                                        'higher_volt_clinique', 'Freq_delta_lab', 'Freq_delta_clinique',
                                                        'Volt_delta_lab', 'Volt_delta_clinique', 'freq_delta_negative_lab',
                                                        'freq_delta_negative_clinique', 'volt_delta_positive_lab',
                                                        'volt_delta_positive_clinique', 'freq_in_range_lab',
                                                        'volt_in_range_lab', 'freq_in_range_clinique', 'volt_in_range_clinique',
                                                        'Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
                                                        'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
                                                        'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18',
                                                        'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23',
                                                        'High_delta_freq_lab', 'High_delta_freq_clinique',
                                                        'High_delta_volt_lab', 'High_delta_volt_clinique',
                                                        'Prev_High_delta_freq_lab', 'Prev_High_delta_freq_clinique',
                                                        'Prev_High_delta_volt_lab', 'Prev_High_delta_volt_clinique'
                                                    ]}

hmmonly = {                              
  "Voltage, Frequency, Mean Threshold, Time Series, HMM":['freq_a', 'freq_b', 'volt_a', 'volt_b',
                                                        'higher_freq_lab', 'higher_freq_clinique', 'higher_volt_lab',
                                                        'higher_volt_clinique', 'Freq_delta_lab', 'Freq_delta_clinique',
                                                        'Volt_delta_lab', 'Volt_delta_clinique', 'freq_delta_negative_lab',
                                                        'freq_delta_negative_clinique', 'volt_delta_positive_lab',
                                                        'volt_delta_positive_clinique', 'freq_in_range_lab',
                                                        'volt_in_range_lab', 'freq_in_range_clinique', 'volt_in_range_clinique',
                                                        'Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6',
                                                        'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12',
                                                        'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18',
                                                        'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23',
                                                        'High_delta_freq_lab', 'High_delta_freq_clinique',
                                                        'High_delta_volt_lab', 'High_delta_volt_clinique',
                                                        'Prev_High_delta_freq_lab', 'Prev_High_delta_freq_clinique',
                                                        'Prev_High_delta_volt_lab', 'Prev_High_delta_volt_clinique'
                                                    ]}
    

# Define hyperparameter grids
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2],
    'eval_metric': ['logloss'],
    'use_label_encoder': [False]
}

def tune_hyperparameters(X_train, y_train, model_type='rf', cv_folds=3, n_iter=20):
    """
    Perform hyperparameter tuning using RandomizedSearchCV
    """
    print(f"Tuning hyperparameters for {model_type}...")
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = rf_param_grid
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        param_grid = xgb_param_grid
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Use RandomizedSearchCV for efficiency
    search = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=cv, 
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best {model_type} score: {search.best_score_:.4f}")
    print(f"Best {model_type} params: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_

def run_experiment(train_portion=1.0, tune_hyperparams=False):
    """
    Run the experiment with different training data portions
    train_portion: 1.0 for original LOSO, 0.6 for partial data inclusion
    """
    print(f"\n{'='*60}")
    if train_portion == 1.0:
        print("RUNNING ORIGINAL LOSO (No held-out site data in training)")
    else:
        print(f"RUNNING PARTIAL DATA INCLUSION ({(1-train_portion)*100:.0f}% of held-out site in training)")
    print(f"Hyperparameter tuning: {'ON' if tune_hyperparams else 'OFF'}")
    print(f"{'='*60}")
    
    results = []
    unique_sites = combined_df['Site'].unique()
    
    # Store best parameters for each site and feature group
    best_params_cache = {}
    failed_site = []
    sites_oi = ['Walikale', 'Katwa', 'Ndosho', 'Kitatumba']

    
    for test_site in sites_oi:
        
        print(f"\n========= Leaving out {test_site} for testing =========\n")
        # try: 
        # Get the held-out site data
        held_out_data = combined_df[combined_df['Site'] == test_site].copy()
        other_sites_data = combined_df[combined_df['Site'] != test_site].copy()
        
        if train_portion < 1.0:
            # Split held-out site data
            held_out_shuffled = held_out_data.copy()
            # sample(frac=1, random_state=42).reset_index(drop=True)
            train_size = int(len(held_out_shuffled) * train_portion)
            
            # held_out_train = held_out_shuffled[:train_size]
            test_data = held_out_shuffled[train_size:]
            
            # Combine with other sites for training
            train_data = other_sites_data
            # train_data = pd.concat([other_sites_data, held_out_train], ignore_index=True)
        else:
            # Original LOSO approach
            train_data = other_sites_data
            test_data = held_out_data
        
        # Generate features
        columns_to_compare = [
            ('freq_a', 'freq_lab'),
            ('freq_b', 'freq_clinique'),
            ('volt_a', 'volt_lab'),
            ('volt_b', 'volt_clinique')
        ]
        
        train_data_ts, test_data_ts = utils.generate_group3features(
            train_data, test_data,
            columns_to_compare,
            'freq_a', 'freq_b', 'volt_a', 'volt_b'
        )
        train_data_ts_co, test_data_ts_co,hmm_features = utils.generate_group4features(train_data_ts, test_data_ts, 'freq_a', 'freq_b', 'volt_a', 'volt_b')
        print(train_data_ts_co.columns)
        print(test_data_ts_co.columns)
        print(hmm_features)
        
        # for group in ['group_1', 'group_2', 'group_3']:
        for group in feature_sets.keys():
            print(f"RUNNING FOR {group}")
            feature_columns = feature_sets[group]
            if group == "Voltage, Frequency, Mean Threshold, Time Series, HMM":
                feature_columns = feature_columns+hmm_features
            
            train_data_model = train_data_ts_co.copy()
            test_data_model = test_data_ts_co.copy()

            # print(f"The columns for train {train_data_model};",train_data_model.columns)
            print(f"The columns for test {test_site}{len(test_data_model.columns)};",(test_data_model.columns))

            print(f"The hmm feature columns for test {test_site} ; {len(feature_columns)};",(feature_columns))

            missing_cols = set(feature_columns) - set(test_data_model.columns)
            if missing_cols:
                print(f"Warning: These columns are missing in test data: {missing_cols}")
            
            assert set(feature_columns).issubset(set(test_data_model.columns)), \
                f"Feature columns {feature_columns} are not all present in test data columns {test_data_model.columns.tolist()}"


            # Prepare data
            X_train = train_data_model[feature_columns].fillna(train_data_model[feature_columns].mean())
            y_train = train_data_model['Generator_ON']
            X_test = test_data_model[feature_columns].fillna(test_data_model[feature_columns].mean())
            y_test = test_data_model['Generator_ON']
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Handle XGBoost data types for group_3
            if group == "Voltage, Frequency, Mean Threshold, Time Series" or group == "Voltage, Frequency, Mean Threshold, Time Series, HMM":
                xgb_cols = ['Prev_High_delta_freq_lab', 'Prev_High_delta_freq_clinique',
                        'Prev_High_delta_volt_lab', 'Prev_High_delta_volt_clinique']
                for col in xgb_cols:
                    if col in X_train.columns:
                        col_idx = X_train.columns.get_loc(col)
                        X_train_scaled[:, col_idx] = pd.to_numeric(X_train[col], errors='coerce').values
                        X_test_scaled[:, col_idx] = pd.to_numeric(X_test[col], errors='coerce').values
            
            # Define models
            models = {
                "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
            }
            
            # Add tuned or default RF and XGB models
            if tune_hyperparams:
                cache_key = f"{test_site}_{group}"
                
                if f"rf_{cache_key}" not in best_params_cache:
                    rf_tuned, rf_params = tune_hyperparameters(X_train_scaled, y_train, 'rf')
                    best_params_cache[f"rf_{cache_key}"] = (rf_tuned, rf_params)
                else:
                    rf_tuned, rf_params = best_params_cache[f"rf_{cache_key}"]
                
                if f"xgb_{cache_key}" not in best_params_cache:
                    xgb_tuned, xgb_params = tune_hyperparameters(X_train_scaled, y_train, 'xgb')
                    best_params_cache[f"xgb_{cache_key}"] = (xgb_tuned, xgb_params)
                else:
                    xgb_tuned, xgb_params = best_params_cache[f"xgb_{cache_key}"]
                
                models["RandomForest_tuned"] = rf_tuned
                models["XGBoost_tuned"] = xgb_tuned
            else:
                # Use default parameters
                models["RandomForest"] = RandomForestClassifier(random_state=42)
                models["XGBoost"] = XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            
            # Train and evaluate models
            for model_name, model in models.items():
                print(f"Training {model_name} for {group} features...")
                
                # Clone the model to avoid fitting issues
                from sklearn.base import clone
                clf = clone(model)
                clf.fit(X_train_scaled, y_train)
                
                # Make predictions
                train_preds = clf.predict(X_train_scaled)
                test_preds = clf.predict(X_test_scaled)
                
                # Evaluate
                train_metrics = utils.evaluate_model_performance_model(y_train, train_preds)
                test_metrics = utils.evaluate_model_performance_model(y_test, test_preds)
                
                # Calculate generator statistics
                test_data_ts['prediction'] = test_preds
                actual_minutes, predicted_minutes = utils.display_generator_statistics(test_data_ts)
                
                # Get feature importance
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                elif hasattr(clf, 'coef_'):
                    importances = np.abs(clf.coef_).flatten()
                else:
                    importances = np.zeros(len(feature_columns))
                
                top_indices = np.argsort(importances)[::-1][:5]
                top_features = [feature_columns[i] for i in top_indices]
                top_feature_str = ', '.join(top_features)
                
                # Store results
                result = {
                    "experiment_type": "modified loso" if train_portion < 1.0 else "original_loso",
                    "heldout_site": test_site,
                    "model": model_name,
                    "featuregroup": group,
                    "hyperparameter_tuning": tune_hyperparams,
                    "top5_features": top_feature_str,
                    "trainset_size": len(X_train),
                    "testset_size": len(X_test),
                    "train_accuracy": train_metrics["Accuracy"],
                    "test_accuracy": test_metrics["Accuracy"],
                    "precision": test_metrics["Precision"],
                    "recall": test_metrics["Recall"],
                    "f1 score": test_metrics.get("F1_Score", 0),
                    "actual_mins_GENON": actual_minutes,
                    "predicted_mins_GENON": predicted_minutes,
                    "Percentage generator on-time detection error": -(100*(actual_minutes-predicted_minutes))/actual_minutes if actual_minutes > 0 else 0
                }
                results.append(result)
                
                print(f"  {model_name} | Train Acc: {train_metrics['Accuracy']:.3f} | Test Acc: {test_metrics['Accuracy']:.3f}")
        # except: 
        #     failed_site.append(test_site)
        #     continue
    return pd.DataFrame(results), failed_site

print("Starting experiments...")

results_crossval_partialintest_tuning, failed_site = run_experiment(train_portion=0.6, tune_hyperparams=tune_hyperparams)


df = results_crossval_partialintest_tuning
df.columns
savepath = f"{root_path}Generator_ML_Emission_Modeling/Modeling/summer25_results/cross_site_models"

df.to_csv(f"{savepath}/results_crossval.csv", index=False)