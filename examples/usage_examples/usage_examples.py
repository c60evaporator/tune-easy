# %% Usage of MuscleTuning
import parent_import
from muscle_tuning import MuscleTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores

# %% Usage of each estimater's Tuning class
from muscle_tuning import LGBMClassifierTuning
from sklearn.datasets import load_boston
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング実行と結果の可視化 ######
tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス
tuning.plot_first_validation_curve(cv=2)  # 範囲を定めて検証曲線をプロット
tuning.optuna_tuning(cv=2)  # Optunaによるチューニング実行
tuning.plot_search_history()  # スコアの上昇履歴を可視化
tuning.plot_search_map()  # 探索点と評価指標を可視化
tuning.plot_best_learning_curve()  # 学習曲線の可視化
tuning.plot_best_validation_curve()  # 検証曲線の可視化
# %%
