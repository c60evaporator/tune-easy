# %% AllInOneTuning, binary, no argument
import parent_import
from tune_easy import AllInOneTuning
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Select 2 classes
TARGET_VARIALBLE = 'species'  # Target variable name
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Selected explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY)
all_tuner.df_scores

# %% AllInOneTuning, binary, all arguments
import parent_import
from tune_easy import AllInOneTuning
import seaborn as sns
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Select 2 classes
TARGET_VARIALBLE = 'species'  # Target variable name
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Selected explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Set arguments
not_opt_params_svm = {'kernel': 'rbf'}
not_opt_params_xgb = {'objective': 'binary:logistic',
                      'random_state': 42,
                      'booster': 'gbtree',
                      'n_estimators': 100,
                      'use_label_encoder': False}
fit_params_xgb = {'verbose': 0,
                  'eval_metric': 'logloss'}
tuning_params_svm = {'gamma': (0.001, 1000),
                     'C': (0.001, 1000)
                     }
tuning_params_xgb = {'learning_rate': (0.05, 0.3),
                     'min_child_weight': (1, 10),
                     'max_depth': (2, 9),
                     'colsample_bytree': (0.2, 1.0),
                     'subsample': (0.2, 1.0)
                     }
# Run parameter tuning
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY,
                             objective='classification', 
                             scoring='auc_ovo',
                             other_scores=['accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_weighted', 'auc_ovr', 'auc_ovo'],
                             learning_algos=['svm', 'xgboost'], 
                             n_iter={'svm': 50,
                                       'xgboost': 20},
                             cv=3, tuning_algo='optuna', seed=42,
                             estimators={'svm': SVC(),
                                         'xgboost': XGBClassifier()},
                             tuning_params={'svm': tuning_params_svm,
                                            'xgboost': tuning_params_xgb},
                             tuning_kws={'svm': {'not_opt_params': not_opt_params_svm},
                                         'xgboost': {'not_opt_params': not_opt_params_xgb,
                                                     'fit_params': fit_params_xgb}}
                             )
all_tuner.df_scores

# %%