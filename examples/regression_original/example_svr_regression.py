# %% SVR, GridSearch, no argument
import parent_import
from param_tuning import SVMRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()

# %% SVR, RandomSearch, no argument
import parent_import
from param_tuning import SVMRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVR, BayesianOptimization, no argument
import parent_import
from param_tuning import SVMRegressorTuning
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVR, Optuna, no argument
import parent_import
from param_tuning import SVMRegressorTuning
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVR, GridSearch, all arguments
import parent_import
from param_tuning import SVMRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE, cv_group=df_reg['ward_after'].values)
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                           }
tuning_params = {'gamma':[0.001, 0.01, 0.03, 0.1, 0.3, 1, 10],
                 'C': [0.01, 0.1, 0.3, 1, 3, 10],
                 'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.2]
                 }
svr = SVR()
not_opt_params = {'kernel': 'rbf'
                  }
param_scales = {'gamma': 'log',
                'C': 'log',
                'epsilon': 'linear'
                }
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
tuning.plot_first_validation_curve(estimator=svr, validation_curve_params=validation_curve_params,
                                   cv=LeaveOneGroupOut(), seed=43, scoring='neg_mean_absolute_error',
                                   not_opt_params=not_opt_params, param_scales=param_scales,
                                   plot_stats='median', axes=axes
                                   )
tuning.grid_search_tuning(estimator=svr, tuning_params=tuning_params,
                          cv=LeaveOneGroupOut(), seed=43,
                          scoring='neg_mean_absolute_error',
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, grid_kws={'n_jobs': 3}
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'epsilon', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 24)}, heat_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()

# %% SVR, RandomSearch, all arguments
import parent_import
from param_tuning import SVMRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE, cv_group=df_reg['ward_after'].values)
tuning_params = {'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                 'C': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                 'epsilon': [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
                 }
svr = SVR()
not_opt_params = {'kernel': 'rbf'
                  }
param_scales = {'gamma': 'log',
                'C': 'log',
                'epsilon': 'linear'
                }
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                           }
tuning.random_search_tuning(estimator=svr, tuning_params=tuning_params,
                          cv=LeaveOneGroupOut(), seed=43,
                          scoring='neg_mean_absolute_error', n_iter=200,
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, rand_kws={'n_jobs': 3}
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'epsilon', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 18)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()

# %% SVR, BayesianOptimization, all arguments
import parent_import
from param_tuning import SVMRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE, cv_group=df_reg['ward_after'].values)
tuning_params = {'gamma':(0.001, 10),
                 'C': (0.01, 10),
                 'epsilon': (0, 0.2)
                 }
svr = SVR()
not_opt_params = {'kernel': 'rbf'
                  }
param_scales = {'gamma': 'log',
                'C': 'log',
                'epsilon': 'linear'
                }
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                           }
tuning.bayes_opt_tuning(estimator=svr, tuning_params=tuning_params,
                          cv=LeaveOneGroupOut(), seed=43,
                          scoring='neg_mean_absolute_error', n_iter=150,
                          init_points=15, acq='ucb',
                          not_opt_params=not_opt_params, int_params=[], param_scales=param_scales,
                          mlflow_logging=None
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'epsilon', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 18)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)

# %% SVR, Optuna, all arguments
import parent_import
from param_tuning import SVMRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import optuna
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
tuning = SVMRegressorTuning(X, y, USE_EXPLATATORY, y_colname=OBJECTIVE_VARIABLE, cv_group=df_reg['ward_after'].values)
tuning_params = {'gamma':(0.001, 10),
                 'C': (0.01, 10),
                 'epsilon': (0, 0.2)
                 }
svr = SVR()
not_opt_params = {'kernel': 'rbf'
                  }
param_scales = {'gamma': 'log',
                'C': 'log',
                'epsilon': 'linear'
                }
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                           'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                           }
tuning.optuna_tuning(estimator=svr, tuning_params=tuning_params,
                     cv=LeaveOneGroupOut(), seed=43,
                     scoring='neg_mean_absolute_error', n_trials=200,
                     study_kws={'sampler': optuna.samplers.CmaEsSampler()},
                     optimize_kws={'show_progress_bar': True},
                     not_opt_params=not_opt_params, int_params=[], param_scales=param_scales,
                     mlflow_logging=None
                     )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'epsilon', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 18)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()
# %%