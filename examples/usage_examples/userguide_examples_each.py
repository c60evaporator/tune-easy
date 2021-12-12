# %% 0.1.1. Feature selection
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
# Load dataset
OBJECTIVE_VARIABLE = 'price'
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIABLE].values  # Objective variable
X_all = california_housing[fetch_california_housing().feature_names].values  # Feature values
# Feature selection
selector = RFE(RandomForestRegressor(random_state=42), n_features_to_select=5)
selector.fit(X_all, y)
print(fetch_california_housing().feature_names)
print(selector.get_support())

# %% 0.1.2. Load dataset with selected explanatory variables
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIABLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIABLE].values  # Objective variable
X = california_housing[USE_EXPLANATORY].values  # Explanatory variables

# %% 0.2. Confirm validation score before tuning


# %% 0.3. Initialize tuning class
import parent_import
from muscle_tuning import LGBMRegressorTuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)

# %% 1. Select validation score
SCORING = 'neg_mean_squared_error'  # RMSE

# %% 2. Select parameter range using validation curve
from sklearn.model_selection import KFold
# Parameter range for validation curve
VALIDATION_CURVE_PARAMS = {'reg_alpha': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                           'reg_lambda': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 192, 256],
                           'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                           'min_child_samples': [0, 2, 5, 10, 20, 30, 50, 70, 100]
                           }
# Plot validation curve
tuning.plot_first_validation_curve(validation_curve_params=VALIDATION_CURVE_PARAMS,
                                   scoring=SCORING,
                                   cv=KFold(n_splits=5, shuffle=True, random_state=42)
                                   )

# %% 4.1 Select cross validation instance
CV = KFold(n_splits=5, shuffle=True, random_state=42)

# %% 4.2.1 Calculate validation score before tuning
from lightgbm import LGBMRegressor
from seaborn_analyzer import cross_val_score_eval_set
import numpy as np
# Fit parameters passed to estimator.fit()
FIT_PARAMS = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'rmse',
              'eval_set': [(X, y)]
              }
# Parameters not used in optimization
NOT_OPT_PARAMS = {'objective': 'regression',
                  'random_state': 42,
                  'boosting_type': 'gbdt',
                  'n_estimators': 10000
                  }
# Make estimator instance
lgbmr = LGBMRegressor(**NOT_OPT_PARAMS)
# Calculate validation score
scores = cross_val_score_eval_set('test',  # How to choose "eval_set" data
        lgbmr, X, y,  # Input data
        scoring=SCORING,  # Validation score selected in section 1
        cv=CV,  # Cross validation instance selected in section 4.2
        fit_params=FIT_PARAMS  # Fit parameters passed to estimator.fit()
        )
print(np.mean(scores))

# %% 4.2.2 Visualize estimator before tuning
from seaborn_analyzer import regplot
california_housing['price'] = y
regplot.regression_pred_true(lgbmr,
                             x=tuning.x_colnames,
                             y='price',
                             data=california_housing,
                             scores='mse',
                             cv=CV,
                             fit_params=FIT_PARAMS,
                             eval_set_selection='test'
                             )

# %% 4.3 Execute parameter tuning
# Select parameter range of optuna
TUNING_PARAMS = {'reg_alpha': (0.0001, 0.1),
                 'reg_lambda': (0.0001, 0.1),
                 'num_leaves': (2, 50),
                 'colsample_bytree': (0.4, 1.0),
                 'subsample': (0.4, 1.0),
                 'subsample_freq': (0, 7),
                 'min_child_samples': (0, 50)
                 }
# Execute parameter tuning
best_params, best_score = tuning.optuna_tuning(scoring=SCORING,
                                               tuning_params=TUNING_PARAMS,
                                               cv=CV,
                                               not_opt_params=NOT_OPT_PARAMS,
                                               fit_params=FIT_PARAMS
                                               )
print(f'Best parameters\n{best_params}\n')  # Optimized parameters
print(f'Not tuned parameters\n{tuning.not_opt_params}\n')  # Parameters not used in optimization
print(f'Best score\n{best_score}\n')  # Best score in optimized parameters
print(f'Elapsed time\n{tuning.elapsed_time}\n')  # Elapsed time

# %% 5.1 Plot search history
tuning.plot_search_history()

# %% 5.2 Visualize relationship between parameters and validation score
tuning.plot_search_map()

# %% 5.3 Plot learning curve
tuning.plot_best_learning_curve()

# %% 5.4 Plot validation curve
tuning.plot_best_validation_curve()

# %% 6.1 Retain optimized estimator
from seaborn_analyzer import cross_val_score_eval_set
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = LGBMRegressor(**params_after)
# Calculate validation score
scores = cross_val_score_eval_set('test',  # How to choose "eval_set" data
        best_estimator, X, y,
        scoring=tuning.scoring,  # Validation score selected in section 1
        cv=tuning.cv,  # Cross validation instance selected in section 4.2
        fit_params=tuning.fit_params  # Fit parameters passed to estimator.fit()
        )
print(np.mean(scores))

# %% 6.2 Visualize estimator after tuning
from seaborn_analyzer import regplot
regplot.regression_pred_true(best_estimator,
                             x=tuning.x_colnames,
                             y='price',
                             data=california_housing,
                             scores='mse',
                             cv=tuning.cv,
                             fit_params=tuning.fit_params,
                             eval_set_selection='test'
                             )

# %% Usage of MLflow logging
import parent_import
from muscle_tuning import E
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# Load dataset
OBJECTIVE_VARIABLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIABLE].values  # Objective variable
X = california_housing[USE_EXPLANATORY].values  # Explanatory variables
# Optimization with MLflow logging
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)  # Make tuning instance
tuning.optuna_tuning(cv=2, mlflow_logging='inside')  # Run tuning with MLflow logging
# %%