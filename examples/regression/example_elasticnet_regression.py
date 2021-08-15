# %% ElasticNet, GridSearch, no argument
import parent_import
from param_tuning import ElasticNetTuning
import pandas as pd
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = ElasticNetTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()

# %% ElasticNet, RandomSearch, no argument
import parent_import
from param_tuning import ElasticNetTuning
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = ElasticNetTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% ElasticNet, BayesianOptimization, no argument
import parent_import
from param_tuning import ElasticNetTuning
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = ElasticNetTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% ElasticNet, Optuna, no argument
import parent_import
from param_tuning import ElasticNetTuning
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = ElasticNetTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% ElasticNet, GridSearch, all arguments
import parent_import
from param_tuning import ElasticNetTuning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = ElasticNetTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG, cv_group=df_reg['ward_after'].values)
validation_curve_params = {'alpha': [0, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
                           'l1_ratio': [0, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 0.97, 0.99, 1]
                           }
tuning_params = {'alpha': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                 'l1_ratio': [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.9, 0.97, 1]
                 }
er = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet())])
not_opt_params = {}
param_scales = {'alpha': 'log',
                'l1_ratio': 'linear'
                }
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
tuning.plot_first_validation_curve(estimator=er, validation_curve_params=validation_curve_params,
                                   cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42, scoring='neg_mean_squared_error',
                                   not_opt_params=not_opt_params, param_scales=param_scales,
                                   plot_stats='median', axes=axes
                                   )
tuning.grid_search_tuning(estimator=er, tuning_params=tuning_params,
                          cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                          scoring='neg_mean_squared_error',
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, grid_kws={'n_jobs': 3}
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['alpha', 'l1_ratio'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 5)}, heat_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='mean', ax=ax)
fig, axes = plt.subplots(1, 2, figsize=(15, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='mean', axes=axes)
tuning.plot_param_importances()

# %% ElasticNet, Optuna, all arguments
import parent_import
from param_tuning import ElasticNetTuning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import optuna
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = ElasticNetTuning(X, y, USE_EXPLANATORY_REG, y_colname=OBJECTIVE_VARIALBLE_REG, cv_group=df_reg['ward_after'].values)
tuning_params = {'alpha':(0.0001, 1),
                 'l1_ratio': (0, 1)
                 }
er = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet())])
not_opt_params = {}
param_scales = {'alpha': 'log',
                'l1_ratio': 'linear'
                }
validation_curve_params = {'alpha': [0, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
                           'l1_ratio': [0, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 0.97, 0.99, 1]
                           }
tuning.optuna_tuning(estimator=er, tuning_params=tuning_params,
                     cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                     scoring='neg_mean_squared_error', n_trials=40,
                     study_kws={'sampler': optuna.samplers.TPESampler(seed=42)},
                     optimize_kws={'show_progress_bar': True},
                     not_opt_params=not_opt_params, int_params=[], param_scales=param_scales,
                     mlflow_logging=None
                     )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['alpha', 'l1_ratio'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 5)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='mean', ax=ax)
fig, axes = plt.subplots(1, 2, figsize=(15, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='mean', axes=axes)
tuning.plot_param_importances()
# %%
