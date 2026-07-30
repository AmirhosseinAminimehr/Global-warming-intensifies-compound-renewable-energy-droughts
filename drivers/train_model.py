import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor

prepared_data_file = "prepared_data_CorrectedFormula.pkl"
trained_models_file = "trained_models_global_Pos_Neg_CorrectedFormula.pkl"

print("--- Training Machine Learning Models ---")

try:
    mean_df = pd.read_pickle(prepared_data_file)

except FileNotFoundError:
    print(f"'{prepared_data_file}' not found.")
    exit()

input_vars = ['diff_rsds_mean', 'diff_uas_mean', 'diff_vas_mean', 'warming_level', 'lon', 'lat']
output_vars = [
    'ratio_intensity_mean',
    'ratio_duration_mean',
    'ratio_frequency',
    'ratio_drought_days'
]

global_data = mean_df.dropna(subset=input_vars + output_vars)

if len(global_data) < 20:
    print("Insufficient data for training.")
    exit()

trained_models = {}

print("Training models...")

for output_var in tqdm(output_vars, desc="Outputs"):
    trained_models[output_var] = {}

    for regime, condition in {
        'positive': global_data[output_var] > 1,
        'negative': global_data[output_var] < 1
    }.items():

        subset = global_data[condition]
        # print(len(subset))
        if len(subset) < 10:
            print(f"Skipping {output_var} ({regime})")
            continue

        X = subset[input_vars]
        y = subset[output_var]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.06,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=0.95,
            random_state=42
        )

        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)

        print(f"{output_var} ({regime}) R²: {r2:.2f}")

        trained_models[output_var][regime] = {
            'model': model,
            'r2_score': r2,
            'X_test': X_test,
            'y_test': y_test
        }

print(f"\nSaving models to '{trained_models_file}'...")
with open(trained_models_file, 'wb') as f:
    pickle.dump(trained_models, f)