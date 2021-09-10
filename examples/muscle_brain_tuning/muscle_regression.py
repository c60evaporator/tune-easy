# %% MuscleTuning, regression, no argument
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
df_boston['price'] = y
# Run tuning
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)
kinnikun.df_scores

# %% MuscleTuning, regression, with argumet
import parent_import
from seaborn_analyzer import classplot
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values

NOT_OPT_PARAMS = {'logr__penalty': 'l2', 'logr__solver': 'lbfgs'}
BEST_PARAMS = {'logr__C': 29.13656314177006}
params = {}
params.update(NOT_OPT_PARAMS)
params.update(BEST_PARAMS)
estimator = Pipeline(steps=[('scaler', StandardScaler()), ('logr', LogisticRegression())])
estimator.set_params(**params)

classplot.class_separator_plot(estimator, X, y, x_colnames=USE_EXPLANATORY, cv=5, pair_sigmarange=1.0)

auc = cross_val_score(estimator, X, y, scoring='roc_auc_ovr', cv=KFold(n_splits=5, shuffle=True, random_state=42))
print(auc)
#classplot.roc_plot(estimator, X, y, x_colnames=USE_EXPLANATORY, cv=5)
# %%
