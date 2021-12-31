# %% LogisticRegression, GridSearch, no argument
import parent_import
from tune_easy import LogisticRegressionTuning
import seaborn as sns
iris = sns.load_dataset("iris")
# Load dataset
TARGET_VARIALBLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
tuning = LogisticRegressionTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
