# %% MLFlow　Grid search
import parent_import
from tune_easy import XGBRegressorTuning
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # Target variable
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
# Set arguments
tuning_params = {'learning_rate': [0.01, 0.03, 0.1, 0.3],
                 'min_child_weight': [2, 4, 6, 8],
                 'max_depth': [1, 2, 3, 4],
                 'colsample_bytree': [0.2, 0.5, 0.8, 1.0],
                 'subsample': [0.2, 0.5, 0.8, 1.0]
                 }
fit_params = {'verbose': 0,  # Command line output
              'eval_set':None,  # Eval set for early_stopping_rounds
              }
not_opt_params = {'booster': 'gbtree',
                  'objective': 'reg:squarederror',
                  'random_state': 42,
                  'n_estimators': 100
                 }
xgbr = XGBRegressor()
# Tuning with MLflow logging
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
tuning.grid_search_tuning(estimator=xgbr, mlflow_logging='inside', tuning_params=tuning_params,
                          cv=KFold(n_splits=3, shuffle=True, random_state=42),
                          not_opt_params=not_opt_params, fit_params=fit_params)

# %% MLFlow Random search
import parent_import
from tune_easy import SVMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # Target variable
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
# Tuning with MLflow logging
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
tuning.random_search_tuning(mlflow_logging='inside')

# %% MLFlow実装　BayesianOptimization
import parent_import
from tune_easy import SVMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # Target variable
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
# Tuning with MLflow logging
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
tuning.bayes_opt_tuning(mlflow_logging='inside')

# %% MLFlow　Optuna
import parent_import
from tune_easy import SVMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # Target variable
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
# Tuning with MLflow logging
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
tuning.optuna_tuning(mlflow_logging='inside')

# %% MLFlow　Optuna SQLite
import parent_import
from tune_easy import SVMRegressorTuning
import pandas as pd
import sqlite3
import os
# MLflow settings
DB_PATH = f'{os.getcwd()}/_tracking_uri/mlruns.db'
EXPERIMENT_NAME = 'optuna_regression'  # Experiment name
ARTIFACT_LOCATION = f'{os.getcwd()}/_artifact_location'  # Artifact location
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # Make directory fo tracking server
conn = sqlite3.connect(DB_PATH)  # Make backend DB
tracking_uri = f'sqlite:///{DB_PATH}'  # Tracking uri
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # Target variable
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
# Tuning with MLflow logging
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
tuning.optuna_tuning(mlflow_logging='inside', mlflow_tracking_uri=tracking_uri,
                     mlflow_experiment_name=EXPERIMENT_NAME, mlflow_artifact_location=ARTIFACT_LOCATION)

# %% MLFlow　Optuna outside
import parent_import
from tune_easy import SVMRegressorTuning
import pandas as pd
import mlflow
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # Target variable
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
# Start MLflow run
with mlflow.start_run() as run:
    # Tuning with MLflow logging
    tuning.optuna_tuning(mlflow_logging='outside')
    mlflow.log_param('data_name', 'osaka_metropolis')
    mlflow.log_dict(tuning.tuning_params, 'tuning_params.json')

# %%
