# %% LogisticRegression, GridSearch, no argument
import parent_import
from muscle_tuning import LogisticRegressionTuning
import seaborn as sns
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
tuning = LogisticRegressionTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
# %%
