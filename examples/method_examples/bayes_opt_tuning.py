# %% bayes_opt_tuning(), no argument
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
###### Run bayes_opt_tuning() ######
best_params, best_score = tuning.bayes_opt_tuning()

# %% bayes_opt_tuning(), Set parameter range by 'tuning_params' argument
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
best_params, best_score = tuning.bayes_opt_tuning(tuning_params=BAYES_PARAMS,
                                                  n_iter=75,
                                                  init_points=10)
# %% bayes_opt_tuning(), Set estimator by 'estimator' argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
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
best_params, best_score = tuning.bayes_opt_tuning(estimator=ESTIMATOR,
                                                  tuning_params=BAYES_PARAMS,
                                                  n_iter=75,
                                                  init_points=10)
# %%
