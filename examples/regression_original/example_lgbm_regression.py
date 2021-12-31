# %% LGBMRegressor, GridSearch, no argument
import parent_import
from tune_easy import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% LGBMRegressor, RandomSearch, no argument
import parent_import
from tune_easy import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% LGBMRegressor, BayesianOptimization, no argument
import parent_import
from tune_easy import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% LGBMRegressor, Optuna, no argument
import parent_import
from tune_easy import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% LGBMRegressor, GridSearch, all arguments
import parent_import
from tune_easy import LGBMRegressorTuning
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning_params = {'reg_alpha': [0.0001, 0.003, 0.1],
                 'reg_lambda': [0.0001, 0.1],
                 'num_leaves': [2, 3, 4, 6],
                 'colsample_bytree': [0.4, 0.7, 1.0],
                 'subsample': [0.4, 1.0],
                 'subsample_freq': [0, 7],
                 'min_child_samples': [0, 2, 5, 10]
                 }
lgbmr = LGBMRegressor()
not_opt_params = {'objective': 'regression',  # 最小化させるべき損失関数
                  'random_state': 42,  # 乱数シード
                  'boosting_type': 'gbdt',  # ブースター
                  'n_estimators': 10000  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                  }
fit_params = {'verbose': 0,  # 学習中のコマンドライン出力
              'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
              'eval_metric': 'rmse',  # early_stopping_roundsの評価指標
              'eval_set': [(X, y)]
              }
param_scales = {'reg_alpha': 'log',
                'reg_lambda': 'log',
                'num_leaves': 'linear',
                'colsample_bytree': 'linear',
                'subsample': 'linear',
                'subsample_freq': 'linear',
                'min_child_samples': 'linear'
                }
validation_curve_params = {'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                           'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
                           'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                           'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                           }
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
tuning.plot_first_validation_curve(estimator=lgbmr, validation_curve_params=validation_curve_params,
                                   cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42, scoring='neg_mean_squared_error',
                                   not_opt_params=not_opt_params, param_scales=param_scales,
                                   plot_stats='median', axes=axes, fit_params=fit_params
                                   )
tuning.grid_search_tuning(estimator=lgbmr, tuning_params=tuning_params,
                          cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                          scoring='neg_mean_squared_error',
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, grid_kws={'n_jobs': 3},
                          **fit_params)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['min_child_samples', 'reg_alpha', 'num_leaves', 'colsample_bytree'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(12, 14)}, heat_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 7, figsize=(30, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
tuning.plot_feature_importances(ax=ax)

# %% LGBMRegressor, Optuna, all arguments
import parent_import
from tune_easy import LGBMRegressorTuning
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import optuna
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning_params = {'reg_alpha': (0.0001, 0.1),
                 'reg_lambda': (0.0001, 0.1),
                 'num_leaves': (2, 6),
                 'colsample_bytree': (0.4, 1.0),
                 'subsample': (0.4, 1.0),
                 'subsample_freq': (0, 7),
                 'min_child_samples': (0, 10)
                 }
lgbmr = LGBMRegressor()
not_opt_params = {'objective': 'regression',  # 最小化させるべき損失関数
                  'random_state': 42,  # 乱数シード
                  'boosting_type': 'gbdt',  # ブースター
                  'n_estimators': 10000  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                  }
fit_params = {'verbose': 0,  # 学習中のコマンドライン出力
              'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
              'eval_metric': 'rmse',  # early_stopping_roundsの評価指標
              'eval_set': [(X, y)]
              }
param_scales = {'reg_alpha': 'log',
                'reg_lambda': 'log',
                'num_leaves': 'linear',
                'colsample_bytree': 'linear',
                'subsample': 'linear',
                'subsample_freq': 'linear',
                'min_child_samples': 'linear'
                }
validation_curve_params = {'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                           'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
                           'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                           'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                           }
int_params = ['num_leaves', 'subsample_freq', 'min_child_samples']
tuning.optuna_tuning(estimator=lgbmr, tuning_params=tuning_params,
                     cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                     scoring='neg_mean_squared_error', n_trials=400,
                     study_kws={'sampler': optuna.samplers.TPESampler(seed=42)},
                     optimize_kws={'show_progress_bar': True},
                     not_opt_params=not_opt_params, int_params=int_params, param_scales=param_scales,
                     mlflow_logging=None,
                     **fit_params
                     )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['min_child_samples', 'reg_alpha', 'num_leaves', 'colsample_bytree'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(20, 15)}, heat_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 7, figsize=(30, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
tuning.plot_feature_importances(ax=ax)
# %%
