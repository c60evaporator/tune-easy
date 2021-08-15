# %% RandomForest, GridSearch, no argument
import parent_import
from param_tuning import RFRegressorTuning
import pandas as pd
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% RandomForest, RandomSearch, no argument
import parent_import
from param_tuning import RFRegressorTuning
import pandas as pd
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% RandomForest, BayesianOptimization, no argument
import parent_import
from param_tuning import RFRegressorTuning
import pandas as pd
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% RandomForest, Optuna, no argument
import parent_import
from param_tuning import RFRegressorTuning
import pandas as pd
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()