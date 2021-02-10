from xgb_param_tuning import XGBRegressorTuning
import xgb_tuning
from xgb_validation import XGBRegressorValidation
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 結果出力先
OUTPUT_DIR = f"{os.getenv('HOMEDRIVE')}{os.getenv('HOMEPATH')}\Desktop"
# 最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
SCORING = 'r2'
# パラメータ最適化の手法(Grid, Random, Bayes, Optuna)
PARAM_TUNING_METHODS = ['Bayes']
# 最適化で使用する乱数シード一覧
SEEDS = [42]

# 使用するフィールド
KEY_VALUE = 'ward_before'  # キー列
OBJECTIVE_VARIALBLE = 'approval_rate'  # 目的変数
EXPLANATORY_VALIABLES = ['1_over60', '2_between_30to60', '3_male_ratio',
    '4_required_time', '5_household_member', '6_income']  # 説明変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio',
    '5_household_member', 'latitude']  # 使用する説明変数


# 現在時刻
dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
# データ読込
df = pd.read_csv(f'./osaka_metropolis_english.csv')
# 目的変数と説明変数を取得（pandasではなくndarrayに変換）
y = df[[OBJECTIVE_VARIALBLE]].values
X = df[USE_EXPLANATORY].values

# パラメータ最適化クラス (旧)
tuning_old = XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)

# パラメータ最適化クラス (新)
tuning_new = xgb_tuning.XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)

best_params_old, best_score_old, feature_importance_old, elapsed_time_old = tuning_old.grid_search_tuning()
#fit_params = {'early_stopping_rounds': 50, "eval_set": [('a', 'b')]}
#pipe = Pipeline([("scaler", StandardScaler()), ("xgb", xgb.XGBRegressor())])
best_params_new, best_score_new, elapsed_time_new = tuning_new.grid_search_tuning()
print(best_params_old)
print(best_params_new)
print(best_score_old)
print(best_score_new)
print(elapsed_time_old)
print(elapsed_time_new)