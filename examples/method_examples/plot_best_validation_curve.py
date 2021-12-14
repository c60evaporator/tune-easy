# %% plot_best_validation_curve(), no argument
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
best_params, best_score = tuning.optuna_tuning()
###### Run plot_best_validation_curve() ######
tuning.plot_best_validation_curve()

# %% plot_best_validation_curve(), Set parameter range by 'validation_curve_params' argument
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
best_params, best_score = tuning.optuna_tuning()
# Set 'validation_curve_params' argument
VALIDATION_CURVE_PARAMS = {'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64],
                           'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
                           'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
                           'min_child_samples': [0, 5, 10, 20, 30, 50]
                           }
###### plot_best_validation_curve() ######
tuning.plot_best_validation_curve(validation_curve_params=VALIDATION_CURVE_PARAMS)

# %%
