# %% random_search_tuning(), no argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)
###### Run random_search_tuning() ######
best_params, best_score = tuning.random_search_tuning()

# %% random_search_tuning(), Set parameter range by 'tuning_params' argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)
# Set 'tuning_params' argument
CV_PARAMS_RANDOM = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                    'max_features': [2, 3, 4, 5],
                    'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                    }
###### Run random_search_tuning() ######
best_params, best_score = tuning.random_search_tuning(tuning_params=CV_PARAMS_RANDOM,
                                                      n_iter=160)

# %% random_search_tuning(), Set estimator by 'estimator' argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)
# Set 'estimator' argument
ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
# Set 'tuning_params' argument
CV_PARAMS_RANDOM = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                    'max_features': [2, 3, 4, 5],
                    'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                    }
###### Run grid_search_tuning() ######
best_params, best_score = tuning.random_search_tuning(estimator=ESTIMATOR,
                                                      tuning_params=CV_PARAMS_RANDOM,
                                                      n_iter=160)

# %%
