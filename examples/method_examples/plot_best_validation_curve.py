# %% plot_best_validation_curve(), no argument
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
###### Run plot_best_validation_curve() ######
tuning.plot_best_validation_curve()

# %% plot_best_validation_curve(), Set parameter range by 'validation_curve_params' argument
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
