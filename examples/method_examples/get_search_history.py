# %% get_search_history(), no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# Load dataset
OBJECTIVE_VARIABLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIABLE].values 
X = california_housing[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
best_params, best_score = tuning.optuna_tuning()
###### Run get_search_history() ######
df_history = tuning.get_search_history()
df_history
# %%
