# %% optuna_tuning(), no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # Objective variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
###### Run optuna_tuning() ######
best_params, best_score = tuning.optuna_tuning()

# %% optuna_tuning(), Set parameter range by 'tuning_params' argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # Objective variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
# Set 'tuning_params' argument
BAYES_PARAMS = {'reg_alpha': (0.001, 0.1),
                'reg_lambda': (0.001, 0.1),
                'num_leaves': (2, 50),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.4, 0.8),
                'subsample_freq': (0, 5),
                'min_child_samples': (0, 20)
                }
###### Run optuna_tuning() ######
best_params, best_score = tuning.optuna_tuning(tuning_params=BAYES_PARAMS,
                                               n_trials=200,
                                               )
# %% optuna_tuning(), Set estimator by 'estimator' argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # Objective variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
# Set 'estimator' argument
ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("lgbmr", LGBMRegressor())])
# Set 'tuning_params' argument
BAYES_PARAMS = {'reg_alpha': (0.001, 0.1),
                'reg_lambda': (0.001, 0.1),
                'num_leaves': (2, 50),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.4, 0.8),
                'subsample_freq': (0, 5),
                'min_child_samples': (0, 20)
                }
###### Run bayes_opt_tuning() ######
best_params, best_score = tuning.optuna_tuning(estimator=ESTIMATOR,
                                               tuning_params=BAYES_PARAMS,
                                               n_trials=150)
# %%