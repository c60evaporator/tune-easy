# %% MuscleTuning, regression, no argument
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# Load dataset
TARGET_VARIALBLE = 'price'  # Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Target variable
# Run tuning
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)
kinnikun.df_scores

# %% MuscleTuning, regression, with argumets
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from xgboost import XGBRegressor
# Load dataset
TARGET_VARIALBLE = 'price'  # Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Target variable
# Set arguments
not_opt_params_svr = {'kernel': 'rbf'}
not_opt_params_xgb = {'objective': 'reg:squarederror',
                      'random_state': 42,
                      'booster': 'gbtree',
                      'n_estimators': 100,
                      'use_label_encoder': False}
fit_params_xgb = {'verbose': 0,
                  'eval_metric': 'rmse'}
tuning_params_svr = {'gamma': (0.001, 1000),
                     'C': (0.001, 1000),
                     'epsilon': (0, 0.3)
                     }
tuning_params_xgb = {'learning_rate': (0.05, 0.3),
                     'min_child_weight': (1, 10),
                     'max_depth': (2, 9),
                     'colsample_bytree': (0.2, 1.0),
                     'subsample': (0.2, 1.0)
                     }
# Run tuning
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY,
                             objective='regression',
                             scoring='mae',
                             other_scores=['rmse', 'mae', 'mape', 'r2'],
                             learning_algos=['svr', 'xgboost'], 
                             n_iter={'svr': 50,
                                     'xgboost': 20},
                             cv=3, tuning_algo='optuna', seed=42,
                             estimators={'svr': SVR(),
                                         'xgboost': XGBRegressor()},
                             tuning_params={'svr': tuning_params_svr,
                                            'xgboost': tuning_params_xgb},
                             tuning_kws={'svr': {'not_opt_params': not_opt_params_svr},
                                         'xgboost': {'not_opt_params': not_opt_params_xgb,
                                                     'fit_params': fit_params_xgb}}
                             )
kinnikun.df_scores

# %%