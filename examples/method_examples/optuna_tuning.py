# %% optuna_tuning(), no argument
import parent_import
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
###### Run optuna_tuning() ######
best_params, best_score = tuning.optuna_tuning()

# %% optuna_tuning(), Set parameter range by 'tuning_params' argument
import parent_import
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
# Load dataset
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
                                               n_trials=2)
# %%