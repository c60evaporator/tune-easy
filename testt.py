# %%
from muscle_tuning import MuscleTuning
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import seaborn as sns

# Load dataset
USE_EXPLANATORY = ['NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
df_boston['price'] = y

# iris = sns.load_dataset("iris")
# OBJECTIVE_VARIALBLE = 'species'  # 目的変数
# USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
# y = iris[OBJECTIVE_VARIALBLE].values
# X = iris[USE_EXPLANATORY].values

kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)

# %%
