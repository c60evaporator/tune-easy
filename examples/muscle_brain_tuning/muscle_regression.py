# %% MuscleTuning, regression, no argument
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIALBLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Objective variable
# Run tuning
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores

# %% MuscleTuning, regression, with argumets
import parent_import
from seaborn_analyzer import classplot
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
# Load dataset
iris = sns.load_dataset("iris")
OBJECTIVE_VARIABLE = 'species'  # Objective variable name
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Selected explanatory variables
y = iris[OBJECTIVE_VARIABLE].values  # Objective variable
X = iris[USE_EXPLANATORY].values  # Explanatory variables

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
