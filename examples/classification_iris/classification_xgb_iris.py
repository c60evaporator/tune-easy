# %% XGBClassifier, GridSearch, no argument
import parent_import
from muscle_tuning import XGBClassifierTuning
from sklearn.model_selection import KFold
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
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
# Load dataset
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
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
# Load dataset
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
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
# Load dataset
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
tuning = XGBClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.optuna_tuning(cv=KFold(n_splits=3, shuffle=True, random_state=42))
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
