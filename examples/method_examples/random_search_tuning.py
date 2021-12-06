# %% random_search_tuning(), no argument
import parent_import
from muscle_tuning import RFRegressorTuning
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
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)
###### Run random_search_tuning() ######
best_params, best_score = tuning.random_search_tuning()

# %% random_search_tuning(), Set parameter range by 'tuning_params' argument
import parent_import
from muscle_tuning import RFRegressorTuning
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
from muscle_tuning import RFRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
# Load dataset
OBJECTIVE_VARIABLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIABLE].values 
X = california_housing[USE_EXPLANATORY].values
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
