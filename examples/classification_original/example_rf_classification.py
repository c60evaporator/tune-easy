# %% RFC, GridSearch, no argument
import parent_import
from muscle_tuning import RFClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = RFClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% RFC, RandomSearch, no argument
import parent_import
from muscle_tuning import RFClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = RFClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% RFC, BayesianOptimization, no argument
import parent_import
from muscle_tuning import RFClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = RFClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% RFC, Optuna, no argument
import parent_import
from muscle_tuning import RFClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = RFClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
