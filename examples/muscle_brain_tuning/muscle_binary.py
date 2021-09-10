# %% MuscleTuning, binary, no argument
import parent_import
from muscle_tuning import MuscleTuning
import seaborn as sns

iris = sns.load_dataset("iris")
iris['species'] = iris['species'].map(lambda x: x.replace('virginica', 'setosa'))
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values

kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)
kinnikun.df_scores

# %%
