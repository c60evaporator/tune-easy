# %% RandomForest, GridSearch, no argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = RFRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% RandomForest, RandomSearch, no argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = RFRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% RandomForest, BayesianOptimization, no argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = RFRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% RandomForest, Optuna, no argument
import parent_import
from tune_easy import RFRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv('../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = RFRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_validation_curve()
tuning.plot_best_learning_curve()
tuning.plot_param_importances()
tuning.plot_feature_importances()

# %% ElasticNet, GridSearch, all arguments
import parent_import
from tune_easy import RFRegressorTuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = RFRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
validation_curve_params = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                           'max_features': ['auto', 'sqrt', 'log2'],
                           'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                           'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                           'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                           }
tuning_params = {'n_estimators': [20, 40, 80],
                 'max_features': ['auto', 1, 2],
                 'max_depth': [2, 8],
                 'min_samples_split': [2, 8],
                 'min_samples_leaf': [1, 8]
                 }
rfr = RandomForestRegressor()
not_opt_params = {}
param_scales = {'n_estimators': 'linear',
                'max_features': 'linear',
                'max_depth': 'linear',
                'min_samples_split': 'linear',
                'min_samples_leaf': 'linear'
                }
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
tuning.plot_first_validation_curve(estimator=rfr, validation_curve_params=validation_curve_params,
                                   cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42, scoring='neg_mean_squared_error',
                                   not_opt_params=not_opt_params, param_scales=param_scales,
                                   plot_stats='median', axes=axes
                                   )
tuning.grid_search_tuning(estimator=rfr, tuning_params=tuning_params,
                          cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                          scoring='neg_mean_squared_error',
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, grid_kws={'n_jobs': 3}
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['n_estimators', 'max_features', 'max_depth', 'min_samples_split'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(12, 10)}, heat_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='mean', ax=ax)
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='mean', axes=axes)
tuning.plot_param_importances()

# %% ElasticNet, Optuna, all arguments
import parent_import
from tune_easy import RFRegressorTuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import optuna
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # Target variable
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
# Run parameter tuning
tuning = RFRegressorTuning(X, y, USE_EXPLATATORY, y_colname=TARGET_VARIABLE)
tuning_params = {'n_estimators': (20, 80),
                 'max_features': (1, 40),
                 'max_depth': (2, 8),
                 'min_samples_split': (2, 8),
                 'min_samples_leaf': (1, 8)
                 }
rfr = RandomForestRegressor()
not_opt_params = {}
param_scales = {'n_estimators': 'linear',
                'max_features': 'linear',
                'max_depth': 'linear',
                'min_samples_split': 'linear',
                'min_samples_leaf': 'linear'
                }
validation_curve_params = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                           'max_features': ['auto', 'sqrt', 'log2'],
                           'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                           'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                           'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                           }
tuning.optuna_tuning(estimator=rfr, tuning_params=tuning_params,
                     cv=KFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                     scoring='neg_mean_squared_error', n_trials=40,
                     study_kws={'sampler': optuna.samplers.TPESampler(seed=42)},
                     optimize_kws={'show_progress_bar': True},
                     not_opt_params=not_opt_params,
                     int_params=['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
                     param_scales=param_scales,
                     mlflow_logging=None
                     )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['n_estimators', 'max_features', 'max_depth', 'min_samples_split'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(20, 15)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='mean', ax=ax)
fig, axes = plt.subplots(1, 5, figsize=(20, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='mean', axes=axes)
tuning.plot_param_importances()
# %%
