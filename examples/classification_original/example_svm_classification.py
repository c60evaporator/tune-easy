# %% SVC, GridSearch, no argument
import parent_import
from tune_easy import SVMClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.plot_first_validation_curve()
tuning.grid_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVC, RandomSearch, no argument
import parent_import
from tune_easy import SVMClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.random_search_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVC, BayesianOptimization, no argument
import parent_import
from tune_easy import SVMClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning.bayes_opt_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVC, Optuna, no argument
import parent_import
from tune_easy import SVMClassifierTuning
import pandas as pd
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
best, not_opt, best_score, elapsed = tuning.optuna_tuning()
tuning.plot_search_history()
tuning.plot_search_map()
tuning.plot_best_learning_curve()
tuning.plot_best_validation_curve()
tuning.plot_param_importances()

# %% SVC, GridSearch, all arguments
import parent_import
from tune_easy import SVMClassifierTuning
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000],
                           'C': [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000]
                           }
tuning_params = {'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                 'C': [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
                 }
svc = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
not_opt_params = {'kernel': 'rbf',
                  'probability': True,
                  'random_state': 42
                  }
param_scales = {'gamma': 'log',
                'C': 'log'
                }
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
tuning.plot_first_validation_curve(estimator=svc, validation_curve_params=validation_curve_params,
                                   cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), seed=42, scoring='neg_log_loss',
                                   not_opt_params=not_opt_params, param_scales=param_scales,
                                   plot_stats='mean', axes=axes
                                   )
tuning.grid_search_tuning(estimator=svc, tuning_params=tuning_params,
                          cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                          scoring='neg_log_loss',
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, grid_kws={'n_jobs': 3}
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 5)}, heat_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()

# %% SVC, RandomSearch, all arguments
import parent_import
from tune_easy import SVMClassifierTuning
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning_params = {'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                 'C': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                 }
svc = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
not_opt_params = {'kernel': 'rbf',
                  'probability': True,
                  'random_state': 42
                  }
param_scales = {'gamma': 'log',
                'C': 'log'
                }
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000],
                           'C': [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000]
                           }
tuning.random_search_tuning(estimator=svc, tuning_params=tuning_params,
                          cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                          scoring='neg_log_loss', n_iter=50,
                          not_opt_params=not_opt_params, param_scales=param_scales,
                          mlflow_logging=None, rand_kws={'n_jobs': 3}
                          )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 5)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='mean', ax=ax)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='mean', axes=axes)
tuning.plot_param_importances()

# %% SVC, BayesianOptimization, all arguments
import parent_import
from tune_easy import SVMClassifierTuning
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE)
tuning_params = {'gamma':(0.01, 100),
                 'C': (0.1, 1000)
                 }
svc = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
not_opt_params = {'kernel': 'rbf',
                  'probability': True,
                  'random_state': 42
                  }
param_scales = {'gamma': 'log',
                'C': 'log'
                }
validation_curve_params = {'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                           'C': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                           }
tuning.bayes_opt_tuning(estimator=svc, tuning_params=tuning_params,
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                        scoring='neg_log_loss', n_iter=40,
                        init_points=5, acq='ei',
                        not_opt_params=not_opt_params, int_params=[], param_scales=param_scales,
                        mlflow_logging=None
                        )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 5)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)

# %% SVC, Optuna, all arguments
import parent_import
from tune_easy import SVMClassifierTuning
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import optuna
# Load dataset
df_clf = pd.read_csv('../sample_data/nba_nfl_2.csv')
TARGET_VARIALBLE = 'league'  # Target variable
USE_EXPLANATORY = ['height', 'weight']  # Explanatory variables
y = df_clf[TARGET_VARIALBLE].values
X = df_clf[USE_EXPLANATORY].values
# Run parameter tuning
tuning = SVMClassifierTuning(X, y, USE_EXPLANATORY, y_colname=TARGET_VARIALBLE, cv_group=df_clf['position'].values)
tuning_params = {'gamma':(0.01, 100),
                 'C': (0.1, 1000)
                 }
svc = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
not_opt_params = {'kernel': 'rbf',
                  'probability': True,
                  'random_state': 42
                  }
param_scales = {'gamma': 'log',
                'C': 'log'
                }
validation_curve_params = {'gamma': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000],
                           'C': [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000]
                           }
tuning.optuna_tuning(estimator=svc, tuning_params=tuning_params,
                     cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), seed=42,
                     scoring='neg_log_loss', n_trials=40,
                     study_kws={'sampler': optuna.samplers.TPESampler(seed=42)},
                     optimize_kws={'show_progress_bar': True},
                     not_opt_params=not_opt_params, int_params=[], param_scales=param_scales,
                     mlflow_logging=None
                     )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_search_history(ax=ax, x_axis='time', plot_kws={'color': 'green'})
plt.show()
tuning.plot_search_map(order=['gamma', 'C'],
                       rounddigits_title=4, rank_number=2, rounddigits_score=4,
                       subplot_kws={'figsize':(6, 5)}, scatter_kws={'cmap': 'YlOrBr'})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
tuning.plot_best_learning_curve(plot_stats='median', ax=ax)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
tuning.plot_best_validation_curve(validation_curve_params=validation_curve_params, param_scales=param_scales,
                                  plot_stats='median', axes=axes)
tuning.plot_param_importances()
# %%
