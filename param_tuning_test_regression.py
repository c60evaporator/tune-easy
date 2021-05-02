# %%ライブラリとデータ読込
from svm_tuning import SVMRegressorTuning
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
import matplotlib.pyplot as plt


# 結果出力先
OUTPUT_DIR = f"{os.getenv('HOMEDRIVE')}{os.getenv('HOMEPATH')}\Desktop"
# 最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
SCORING = 'neg_mean_squared_error'
# パラメータ最適化の手法(grid, random, bayes, optuna)
PARAM_TUNING_METHODS = ['bayes']
# 学習器の種類(xgb_old, xgb, xgb_pipe, svm)
LEARNING_METHODS = ['xgb']
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

# %% 検証曲線のプロット
for learning_algo in LEARNING_METHODS:
    if learning_algo == 'xgb':
        tuning_new = xgb_tuning.XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
    elif learning_algo == 'svm':
        tuning_new = SVMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
fig, axes = plt.subplots(5, 1, figsize=(6, 18))
#tuning_new.plot_first_validation_curve(axes=axes)

# %% チューニング実行
def xgb_reg_test_old(tuning_algo):
    # パラメータ最適化クラス (旧)
    tuning_old = XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
    if tuning_algo == 'grid':
        best_params_old, best_score_old, feature_importance_old, elapsed_time_old = tuning_old.grid_search_tuning()
    elif tuning_algo == 'random':
        best_params_old, best_score_old, feature_importance_old, elapsed_time_old = tuning_old.random_search_tuning()
    elif tuning_algo == 'bayes':
        best_params_old, best_score_old, feature_importance_old, elapsed_time_old = tuning_old.bayes_opt_tuning()
    return best_params_old, best_score_old, elapsed_time_old, tuning_old

def xgb_reg_test(tuning_algo):
    # パラメータ最適化クラス (新)
    tuning_new = xgb_tuning.XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE, eval_from_test=True)
    if tuning_algo == 'grid':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.grid_search_tuning()
    elif tuning_algo == 'random':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.random_search_tuning()
    elif tuning_algo == 'bayes':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.bayes_opt_tuning()
    return best_params_new, best_score_new, elapsed_time_new, tuning_new


def xgb_pipe_reg_test(tuning_algo):
    tuning_new = xgb_tuning.XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
    pipe = Pipeline([("scaler", StandardScaler()), ("xgb", xgb.XGBRegressor())])
    not_opt_params = {'xgb__eval_metric': 'rmse',  # データの評価指標
                        'xgb__objective': 'reg:squarederror',  # 最小化させるべき損失関数
                        'xgb__random_state': 42,  # 乱数シード
                        'xgb__booster': 'gbtree',  # ブースター
                        'xgb__n_estimators': 10000  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                        }
    bayes_params = {'learning_rate': (0.1, 0.5),
                        'min_child_weight': (1, 15),
                        'max_depth': (3, 7),
                        'colsample_bytree': (0.5, 1),
                        'subsample': (0.5, 1)
                        }
    fit_params = {'xgb__verbose': 1,  # 学習中のコマンドライン出力
                    'xgb__early_stopping_rounds': 20  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
                    }
    if tuning_algo == 'grid':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.grid_search_tuning(cv_model=pipe)
    elif tuning_algo == 'random':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.random_search_tuning(cv_model=pipe)
    elif tuning_algo == 'bayes':
        fit_params = {'xgb__verbose': 0,'xgb__early_stopping_rounds': 20}
        best_params_new, best_score_new, elapsed_time_new = tuning_new.bayes_opt_tuning(cv_model=pipe)
    return best_params_new, best_score_new, elapsed_time_new, tuning_new

def svm_reg_test(tuning_algo):
    # パラメータ最適化クラス (新)
    tuning_new = SVMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
    if tuning_algo == 'grid':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.grid_search_tuning()
    elif tuning_algo == 'random':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.random_search_tuning()
    elif tuning_algo == 'bayes':
        best_params_new, best_score_new, elapsed_time_new = tuning_new.bayes_opt_tuning()
    return best_params_new, best_score_new, elapsed_time_new, tuning_new


# チューニング実行
result_list = []  # チューニング結果を保持
validation_curve_list = []  # 検証曲線用にデータ保持
for learning_algo in LEARNING_METHODS:
    for tuning_algo in PARAM_TUNING_METHODS:
        if learning_algo == 'xgb_old':
            best_params, best_score, elapsed_time, tuning_instance = xgb_reg_test_old(tuning_algo)
        elif learning_algo == 'xgb':
            best_params, best_score, elapsed_time, tuning_instance = xgb_reg_test(tuning_algo)
        elif learning_algo == 'xgb_pipe':
            best_params, best_score, elapsed_time, tuning_instance = xgb_pipe_reg_test(tuning_algo)
        elif learning_algo == 'svm':
            best_params, best_score, elapsed_time, tuning_instance = svm_reg_test(tuning_algo)
        result = {
            'learning_algo': learning_algo,
            'tuning_algo': tuning_algo,
            'best_score': best_score,
            'elapsed_time': elapsed_time
        }
        result.update({f'best_{k}': v for k, v in best_params.items()})
        result_list.append(result)
        validation_curve_list.append({
            'learning_algo': learning_algo,
            'tuning_algo': tuning_algo,
            'tuning_instance': tuning_instance,
            'best_params': best_params
        })

# 結果表示
df_result = pd.DataFrame(result_list)
print(df_result[['learning_algo', 'tuning_algo', 'best_score', 'elapsed_time']])

# %%学習曲線の表示
for validation_dict in validation_curve_list:
    validation_dict['tuning_instance'].plot_best_learning_curve()

# %%検証曲線の表示
for validation_dict in validation_curve_list:
    fig, axes = plt.subplots(5, 1, figsize=(6, 18))
    validation_dict['tuning_instance'].plot_best_validation_curve(axes=axes)
# %%
