# %% XGBClassifier, GridSearch, no argument
import parent_import
from muscle_tuning import XGBClassifierTuning
from sklearn.model_selection import KFold
import seaborn as sns
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.plot_first_validation_curve(cv=KFold(n_splits=3, shuffle=True, random_state=42))
tuning.grid_search_tuning(cv=KFold(n_splits=3, shuffle=True, random_state=42))
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% XGBClassifier, RandomSearch, no argument
import parent_import
from muscle_tuning import XGBClassifierTuning
from sklearn.model_selection import KFold
import seaborn as sns
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.random_search_tuning(cv=KFold(n_splits=3, shuffle=True, random_state=42))
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% XGBClassifier, BayesianOptimization, no argument
import parent_import
from muscle_tuning import XGBClassifierTuning
from sklearn.model_selection import KFold
import seaborn as sns
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.bayes_opt_tuning(cv=KFold(n_splits=3, shuffle=True, random_state=42))
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% XGBClassifier, Optuna, no argument
import parent_import
from muscle_tuning import XGBClassifierTuning
import seaborn as sns
from sklearn.model_selection import KFold
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
tuning.optuna_tuning(cv=KFold(n_splits=3, shuffle=True, random_state=42))
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
