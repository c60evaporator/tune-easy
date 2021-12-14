# %% plot_feature_importances(), no argument
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
tuning.plot_search_map()
###### Run plot_feature_importances() ######
tuning.plot_feature_importances()

# %%
