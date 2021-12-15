# %% MuscleTuning, binary, no argument
import parent_import
from muscle_tuning import MuscleTuning
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
iris['species'] = iris['species'].map(lambda x: x.replace('versicolor', 'setosa'))
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores

# %%
