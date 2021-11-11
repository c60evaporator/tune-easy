# %% MLFlow実装　グリッドサーチ
import parent_import
from muscle_tuning import XGBRegressorTuning
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning_params = {'learning_rate': [0.01, 0.03, 0.1, 0.3],
                 'min_child_weight': [2, 4, 6, 8],
                 'max_depth': [1, 2, 3, 4],
                 'colsample_bytree': [0.2, 0.5, 0.8, 1.0],
                 'subsample': [0.2, 0.5, 0.8, 1.0]
                 }
fit_params = {'verbose': 0,  # 学習中のコマンドライン出力
              'eval_set':None,  # early_stopping_roundsの評価指標算出用データ
              }
not_opt_params = {'booster': 'gbtree',
                  'objective': 'reg:squarederror',
                  'random_state': 42,
                  'n_estimators': 100
                 }
xgbr = XGBRegressor()
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.grid_search_tuning(estimator=xgbr, mlflow_logging='with', tuning_params=tuning_params,
                          cv=KFold(n_splits=3, shuffle=True, random_state=42),
                          not_opt_params=not_opt_params, fit_params=fit_params)

# %% MLFlow実装　ランダムサーチ
import parent_import
from muscle_tuning import SVMRegressorTuning
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.random_search_tuning(mlflow_logging='with')

# %% MLFlow実装　BayesianOptimization
import parent_import
from muscle_tuning import SVMRegressorTuning
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.bayes_opt_tuning(mlflow_logging='with')

# %% MLFlow実装　Optuna
import parent_import
from muscle_tuning import SVMRegressorTuning
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.optuna_tuning(mlflow_logging='with')

# %% MLFlow実装　グリッドサーチautolog
import parent_import
from muscle_tuning import SVMRegressorTuning
import pandas as pd
import mlflow
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG, cv_group=df_reg['ward_after'].values)
mlflow.sklearn.autolog()
with mlflow.start_run() as run:
    tuning.grid_search_tuning()

# %% MLFlow実装　ランダムサーチautolog
import parent_import
from muscle_tuning import SVMRegressorTuning
import pandas as pd
import mlflow
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG, cv_group=df_reg['ward_after'].values)
mlflow.sklearn.autolog()
with mlflow.start_run() as run:
    tuning.random_search_tuning()