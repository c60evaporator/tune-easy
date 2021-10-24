# %% Usage
import parent_import
from muscle_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = boston[USE_EXPLANATORY].values
y = load_boston().target
###### チューニング実行と評価 ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス
tuning.plot_first_validation_curve()  # 範囲を定めて検証曲線をプロット
tuning.optuna_tuning()  # Optunaによるチューニング実行
tuning.plot_search_history()  # Optuna実行
tuning.plot_search_map()  # 探索点と評価指標を可視化
tuning.plot_best_learning_curve()  # 学習曲線の可視化
tuning.plot_best_validation_curve()  # 検証曲線の可視化