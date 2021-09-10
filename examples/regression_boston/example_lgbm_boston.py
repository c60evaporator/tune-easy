# %% LGBMRegressor, GridSearch, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
OBJECTIVE_VARIALBLE = 'price'  # 目的変数
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
boston = pd.DataFrame(np.column_stack((load_boston().data, load_boston().target)), columns = np.append(load_boston().feature_names, OBJECTIVE_VARIALBLE))
y = boston[OBJECTIVE_VARIALBLE].values
X = boston[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
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
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
OBJECTIVE_VARIALBLE = 'price'  # 目的変数
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
boston = pd.DataFrame(np.column_stack((load_boston().data, load_boston().target)), columns = np.append(load_boston().feature_names, OBJECTIVE_VARIALBLE))
y = boston[OBJECTIVE_VARIALBLE].values
X = boston[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% LGBMRegressor, BayesianOptimization, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
OBJECTIVE_VARIALBLE = 'price'  # 目的変数
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
boston = pd.DataFrame(np.column_stack((load_boston().data, load_boston().target)), columns = np.append(load_boston().feature_names, OBJECTIVE_VARIALBLE))
y = boston[OBJECTIVE_VARIALBLE].values
X = boston[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% LGBMRegressor, Optuna, no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
OBJECTIVE_VARIALBLE = 'price'  # 目的変数
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
boston = pd.DataFrame(np.column_stack((load_boston().data, load_boston().target)), columns = np.append(load_boston().feature_names, OBJECTIVE_VARIALBLE))
y = boston[OBJECTIVE_VARIALBLE].values
X = boston[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
