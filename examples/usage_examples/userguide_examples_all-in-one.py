# %% Classification
import parent_import
from tune_easy import AllInOneTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Only 2 class
TARGET_VARIABLE = 'species'  # Target variable name
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Selected explanatory variables
y = iris[TARGET_VARIABLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
all_tuner.df_scores

# %% Regression
import parent_import
from tune_easy import AllInOneTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
TARGET_VARIABLE = 'price'  # Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIABLE].values  # Target variable
X = california_housing[USE_EXPLANATORY].values  # Explanatory variables
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
all_tuner.df_scores

# %% MLflow
import parent_import
from tune_easy import AllInOneTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
TARGET_VARIABLE = 'price'  # Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIABLE].values  # Target variable
X = california_housing[USE_EXPLANATORY].values  # Explanatory variables
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                             mlflow_logging=True)  # MLflowによるロギング有効化
all_tuner.df_scores
# %%
