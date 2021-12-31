import sys
import os
sys.path.append(os.path.abspath(".."))  # 親ディレクトリからライブラリ読込
import pytest
from tune_easy import SVMRegressorTuning, SVMClassifierTuning, XGBRegressorTuning, RFRegressorTuning
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 回帰用データ読込
TARGET_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
y_reg = df_reg[TARGET_VARIALBLE_REG].values
X_reg = df_reg[USE_EXPLANATORY_REG].values

# 分類用データ読込
TARGET_VARIALBLE_CLS = 'league'  # 目的変数
USE_EXPLANATORY_CLS = ['height', 'weight']  # 説明変数
df_cls = pd.read_csv(f'../sample_data/nba_nfl_2.csv')
y_cls = df_cls[TARGET_VARIALBLE_CLS].values
X_cls = df_cls[USE_EXPLANATORY_CLS].values

###### 正常系：引数なしで実行 ######

def test_svm_reg_grid_noarg():
    tuning = SVMRegressorTuning(X_reg, y_reg, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
    assert tuning.grid_search_tuning()

def test_xgb_reg_grid_noarg():
    tuning = XGBRegressorTuning(X_reg, y_reg, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
    assert tuning.grid_search_tuning()