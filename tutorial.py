# %% 3次元パラメータ(SVR)
from svm_tuning import SVMRegressorTuning
from sklearn.svm import SVR
import pandas as pd
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
params = {'gamma':[0.001, 0.01, 0.1, 1, 10],
          'C': [0.01, 0.1, 1, 10],
          'epsilon': [0, 0.05, 0.1]
          }
#tuning.grid_search_tuning(cv_params=params)
# %% 5次元パラメータ(XGB)
from xgb_tuning import XGBRegressorTuning
from xgboost import XGBRegressor
import pandas as pd
df_reg = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
params = {'learning_rate': [0.1, 0.3, 0.5],  # 過学習のバランス(高いほど過学習寄り、低いほど汎化寄り）別名eta
          'min_child_weight': [1, 5, 15],  # 葉に割り当てるスコアwiの合計の最小値。これを下回った場合、それ以上の分割を行わない
          'max_depth': [3, 5, 7],  # 木の深さの最大値
          'colsample_bytree': [0.5, 0.8, 1.0],  # 列のサブサンプリングを行う比率
          'subsample': [0.5, 0.8, 1.0]  # 木を構築する前にデータのサブサンプリングを行う比率。1 なら全データ使用、0.5なら半分のデータ使用
          }
tuning.grid_search_tuning(cv_params=params)
# %%
