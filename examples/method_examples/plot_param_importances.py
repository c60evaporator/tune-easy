# %% plot_param_importances(), no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']
df_california = pd.DataFrame(fetch_california_housing().data, 
        columns=fetch_california_housing().feature_names)
X = df_california[USE_EXPLANATORY].values
y = fetch_california_housing().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
best_params, best_score = tuning.optuna_tuning()
tuning.plot_search_map()
###### Run plot_param_importances() ######
tuning.plot_param_importances()
# %%
