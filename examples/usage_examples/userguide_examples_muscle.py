# %% Classification
import parent_import
from muscle_tuning import MuscleTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores

# %% Regression
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
TARGET_VARIALBLE = 'price'  # Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Target variable
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores

# %% MLflow
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
TARGET_VARIALBLE = 'price'  # Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Target variable
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                             mlflow_logging=True)  # MLflowによるロギング有効化
kinnikun.df_scores
# %%
