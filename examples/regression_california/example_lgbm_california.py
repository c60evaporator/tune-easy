# %% LGBMRegressor, GridSearch, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIABLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数21000→1000にサンプリング
y = california_housing[OBJECTIVE_VARIABLE].values
X = california_housing[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% LGBMRegressor, RandomSearch, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIABLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数21000→1000にサンプリング
y = california_housing[OBJECTIVE_VARIABLE].values
X = california_housing[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% LGBMRegressor, BayesianOptimization, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIABLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数21000→1000にサンプリング
y = california_housing[OBJECTIVE_VARIABLE].values
X = california_housing[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% LGBMRegressor, Optuna, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIABLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数21000→1000にサンプリング
y = california_housing[OBJECTIVE_VARIABLE].values
X = california_housing[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
