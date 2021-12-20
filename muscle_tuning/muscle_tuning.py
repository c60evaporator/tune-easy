from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import mlflow
from seaborn_analyzer import regplot, classplot
import numbers
import copy
import os

from .linearregression_tuning import LinearRegressionTuning
from .elasticnet_tuning import ElasticNetTuning
from .lgbm_tuning import LGBMClassifierTuning, LGBMRegressorTuning
from .rf_tuning import RFClassifierTuning, RFRegressorTuning
from .svm_tuning import SVMClassifierTuning, SVMRegressorTuning
from .xgb_tuning import XGBClassifierTuning, XGBRegressorTuning
from .logisticregression_tuning import LogisticRegressionTuning


class MuscleTuning():
    SCORING = {'regression': 'rmse',
               'binary': 'logloss',
               'multiclass': 'logloss'
               }
    OTHER_SCORES = {'binary': ['accuracy', 'precision', 'recall', 'f1', 'logloss', 'auc'],
                    'multiclass': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'logloss', 'auc_ovr'],  # 多クラスではprecision, recallは文字列指定不可https://stackoverflow.com/questions/46598301/how-to-compute-precision-recall-and-f1-score-of-an-imbalanced-dataset-for-k-fold
                    'regression': ['rmse', 'mae', 'mape', 'r2']  # RMSLEはpredicted_valueあるいはtrue_valueが負値のときにエラーが出るので注意
                    }
    LEARNING_ALGOS = {'regression': ['linear_regression', 'elasticnet', 'svr', 'randomforest', 'lightgbm'],
                      'binary': ['svm', 'logistic', 'randomforest', 'lightgbm'],
                      'multiclass': ['svm', 'logistic', 'randomforest', 'lightgbm']
                      }
    N_ITER = {'regression': {'svr': 500,
                             'elasticnet': 500,
                             'randomforest': 300, 
                             'lightgbm': 200,
                             'xgboost': 100
                             },
              'binary': {'svm': 500,
                         'logistic': 500,
                         'randomforest': 300, 
                         'lightgbm': 200,
                         'xgboost': 100
                         },
              'multiclass': {'svm': 50,
                             'logistic': 500,
                             'randomforest': 300, 
                             'lightgbm': 200,
                             'xgboost': 100
                            }
              }
    
    _SCORE_RENAME_DICT = {'rmse': 'neg_root_mean_squared_error',
                          'mse': 'neg_mean_squared_error',
                          'mae': 'neg_mean_absolute_error',
                          'rmsle': 'neg_mean_squared_log_error',
                          'mape': 'neg_mean_absolute_percentage_error',
                          'r2': 'r2',
                          'logloss': 'neg_log_loss',
                          'auc': 'roc_auc',
                          'auc_ovr': 'roc_auc_ovr',
                          'auc_ovo': 'roc_auc_ovo',
                          'auc_ovr_weighted': 'roc_auc_ovr_weighted',
                          'auc_ovo_weighted': 'roc_auc_ovo_weighted',
                          'pr_auc': 'average_precision',
                          'accuracy': 'accuracy',
                          'precision': 'precision',
                          'recall': 'recall',
                          'precision_macro': make_scorer(precision_score, average='macro'),
                          'recall_macro': make_scorer(recall_score, average='macro'),
                          'f1': 'f1',
                          'f1_micro': 'f1_micro',
                          'f1_macro': 'f1_macro',
                          'f1_weighted': 'f1_weighted',
                          }
    _SCORE_NEGATIVE = {'rmse': True,
                       'mse': True,
                       'mae': True,
                       'rmsle': True,
                       'mape': True,
                       'r2': False,
                       'logloss': True,
                       'auc': False,
                       'auc_ovr': False,
                       'auc_ovo': False,
                       'auc_ovr_weighted': False,
                       'auc_ovo_weighted': False,
                       'pr_auc': False,
                       'accuracy': False,
                       'precision': False,
                       'recall': False,
                       'precision_macro': False,
                       'recall_macro': False,
                       'f1': False,
                       'f1_micro': False,
                       'f1_macro': False,
                       'f1_weighted': False,
                       }
    _COLOR_LIST = list(colors.TABLEAU_COLORS.values())

    def _reshape_input_data(self, x, y, data, x_colnames, cv_group):
        """
        入力データの形式統一(pd.DataFrame or np.ndarray)
        """
        # dataがpd.DataFrameのとき
        if isinstance(data, pd.DataFrame):
            if not isinstance(x, list):
                raise Exception('`x` argument should be list[str] if `data` is pd.DataFrame')
            if not isinstance(y, str):
                raise Exception('`y` argument should be str if `data` is pd.DataFrame')
            if x_colnames is not None:
                raise Exception('`x_colnames` argument should be None if `data` is pd.DataFrame')
            self.X = data[x].values
            self.y = data[y].values
            self.x_colnames = x
            self.y_colname = y
            self.group_name = cv_group
            if cv_group is not None:  # cv_group指定時
                self.cv_group = data[cv_group].values
                self.data = data[x + [y] + [cv_group]]
            else:
                self.data = data[x + [y]]
            
        # dataがNoneのとき(x, yがnp.ndarray)
        elif data is None:
            if not isinstance(x, np.ndarray):
                raise Exception('`x` argument should be list[str] if `data` is None')
            if not isinstance(y, np.ndarray):
                raise Exception('`y` argument should be np.ndarray if `data` is None')
            self.X = x if len(x.shape) == 2 else x.reshape([x.shape[0], 1])
            self.y = y.ravel()
            # x_colnameとXの整合性確認
            if x_colnames is None:
                self.x_colnames = list(range(self.X.shape[1]))
            elif self.X.shape[1] != len(x_colnames):
                raise Exception('width of X must be equal to length of x_colnames')
            else:
                self.x_colnames = x_colnames
            self.y_colname = 'target_variable'
            self.cv_group = cv_group
            if cv_group is not None:  # cv_group指定時
                self.group_name = 'group'
                self.data = pd.DataFrame(np.column_stack((self.X, self.y, self.cv_group)),
                                     columns=self.x_colnames + [self.y_colname] + [self.group_name])
            else:
                self.data = pd.DataFrame(np.column_stack((self.X, self.y)),
                                     columns=self.x_colnames + [self.y_colname])
        else:
            raise Exception('`data` argument should be pd.DataFrame or None')

    def _initialize(self, x, y, data, x_colnames, cv_group=None):
        """
        プロパティの初期化
        """
        # 入力データの整形
        self.X = None
        self.y = None
        self.data = None
        self.x_colnames = None
        self.y_colname = None
        self.cv_group = None  # GroupKFold, LeaveOneGroupOut用のグルーピング対象データ
        self.group_name = None
        self._reshape_input_data(x, y, data, x_colnames, cv_group)
        # データから判定するプロパティ
        self.objective = None  # タスク ('regression', 'binary', 'multiclass')
        # 定数から読み込むプロパティ
        self.scoring = None  # 最大化するスコア
        self.other_scores = None  # チューニング後に表示するスコア一覧
        self.learning_algos = None  # 比較する学習器の一覧
        self.n_iter = None  # 学習器ごとのチューニング試行数 (グリッドサーチ以外で有効)
        # 引数指定したデフォルト値を読み込むプロパティ
        self.tuning_algo = None  # 最適化に使用したアルゴリズム名('grid', 'random', 'bayes-opt', 'optuna')
        self.cv = None  # クロスバリデーション用インスタンス
        self.seed = None  # 乱数シード
        self.mlflow_logging=False  # MLflowロギング有無
        self.mlflow_tracking_uri=None  # MLflowのTracking URI
        self.mlflow_artifact_location=None  # MLflowのArtifactストレージ
        self.mlflow_experiment_name=None  # MLflowのExperiment名
        # チューニング用クラスのデフォルト値を使用するプロパティ
        self.estimators = None  # 学習器インスタンスのdict
        self.tuning_params = None  # チューニング対象のパラメータとその範囲のdict
        self.tuning_kws = None  # チューニング用メソッドに渡す引数
        # チューニング後に取得するプロパティ
        self.best_scores = {}  # スコア
        self.tuners = {}  # チューニング後のParamTuning継承クラス保持用
        self.estimators_before = {}  # チューニング前の学習器
        self.estimators_after = {}  # チューニング後の学習器
        self.df_scores = None  # 算出したスコアを保持するDataFrame
        self.df_scores_cv = None  # 算出したスコアをクロスバリデーション全て保持するDataFrame
        self.best_learner = None  # 最もスコアの良かった学習器名
    
    def _select_objective(self, objective):
        """
        タスクの判定
        """
        objective_dtype = self.y.dtype
        # 回帰指定時
        if objective == 'regression':
            if isinstance(objective, (int, float)):
                self.objective = 'regression'
            else:
                raise Exception('Target variable should be int or float if `objective` is "regression"')
        # 分類指定時
        elif objective == 'classification':
            n_classes = len(np.unique(self.y))  # 目的変数の固有メンバー数
            if n_classes <= 1:  # 固有メンバー数の下限は2
                raise Exception('Number of unique members of target variable should be bigger than 1')
            elif n_classes == 2:  # 2クラス分類
                self.objective = 'binary'
            elif n_classes <= 20:  # 多クラス分類
                self.objective = 'multiclass'
            else:  # 固有メンバーは20が上限
                raise Exception('Number of unique members of target variable should be less than 20')
        # タスク未指定時
        elif objective is None:
            # float型 or int型のとき、回帰タスクとみなす
            if objective_dtype in [np.int32, np.int64, np.float32, np.float64]:
                self.objective = 'regression'
            # str型 or bool型のとき、分類タスクとみなす
            elif objective_dtype in [np.object, np.bool]:
                n_classes = len(np.unique(self.y))  # 目的変数の固有メンバー数
                if n_classes <= 1:  # 固有メンバー数の下限は2
                    raise Exception('Number of unique members of target variable should be bigger than 1')
                elif n_classes == 2:  # 2クラス分類
                    self.objective = 'binary'
                elif n_classes <= 20:  # 多クラス分類
                    self.objective = 'multiclass'
                else:  # 固有メンバーは20が上限
                    raise Exception('Number of unique members of target variable should be less than 20')
        else:
            raise Exception('`objective` argument should be "regression" or "classification"')

    def _set_property_from_const(self, scoring, other_scores, learning_algos, n_iter):
        """
        未指定時に定数から読み込むプロパティ
        """
        # チューニング用評価指標
        if scoring is None:
            self.scoring = self.SCORING[self.objective]
        else:
            self.scoring = scoring
        # チューニング後に表示するスコアのリスト (NoneならOTHER_SCORESを使用)
        if other_scores is None:
            self.other_scores = self.OTHER_SCORES[self.objective]
        else:
            self.other_scores = other_scores
        # 学習器名称のリスト (NoneならLEARNING_ALGOSを使用)
        if learning_algos is None:
            self.learning_algos = self.LEARNING_ALGOS[self.objective]
        else:
            self.learning_algos = learning_algos
        # 試行数のリスト (グリッドサーチ以外で有効)
        if n_iter is None:
            self.n_iter = self.N_ITER[self.objective]
        else:
            self.n_iter = n_iter
        
    def _set_property_from_arguments(self, cv, tuning_algo, seed, mlflow_logging, mlflow_tracking_uri, mlflow_artifact_location, mlflow_experiment_name):
        """
        未指定時にデフォルト引数から読み込むプロパティ
        """
        # Cross validationを指定
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):# GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')
        if isinstance(cv, numbers.Integral):# int指定時、KFoldで分割
            self.cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        else:
            self.cv = cv
        # tuning_algo
        self.tuning_algo = tuning_algo
        # seed
        self.seed = seed
        # MLflow系
        self.mlflow_logging = mlflow_logging
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_artifact_location = mlflow_artifact_location
        self.mlflow_experiment_name = mlflow_experiment_name
    
    def _set_property_from_algo(self, estimators, tuning_params, tuning_kws):
        """
        未指定時にチューニング用クラスのデフォルト値から読み込むプロパティ
        """
        # 学習器インスタンスのリスト (Noneなら全てNone=チューニング用クラスのデフォルト値を使用)
        if estimators is None:
            self.estimators = {k: None for k in self.learning_algos}
        else:
            self.estimators = estimators
        # チューニング対象パラメータ範囲のdict (Noneなら全てNone=チューニング用クラスのデフォルト値を使用)
        if tuning_params is None:
            self.tuning_params = {k: None for k in self.learning_algos}
        else:
            self.tuning_params = tuning_params
        # チューニング用メソッドに渡す引数 (Noneなら空のdict=チューニング用クラスのデフォルト値を使用)
        if tuning_kws is None:
            self.tuning_kws = {k: {} for k in self.learning_algos}
        else:
            self.tuning_kws = tuning_kws

    def _run_tuning(self, tuner, estimator, tuning_params, n_iter, tuning_kws, mlflow_logging):
        """
        チューニング用メソッド実行
        """
        # グリッドサーチ
        if self.tuning_algo == 'grid':
            tuner.grid_search_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self._SCORE_RENAME_DICT[self.scoring],
                                    mlflow_logging=mlflow_logging,
                                    **tuning_kws
                                    )
        # ランダムサーチ
        elif self.tuning_algo == 'random':
            tuner.grid_search_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self._SCORE_RENAME_DICT[self.scoring],
                                    mlflow_logging=mlflow_logging,
                                    n_iter=n_iter,
                                    **tuning_kws
                                    )
        # BayesianOptimization
        elif self.tuning_algo == 'bayes-opt':
            tuner.bayes_opt_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self._SCORE_RENAME_DICT[self.scoring],
                                    mlflow_logging=mlflow_logging,
                                    n_iter=n_iter,
                                    **tuning_kws
                                    )
        # Optuna
        elif self.tuning_algo == 'optuna':
            tuner.optuna_tuning(estimator=estimator,
                                tuning_params=tuning_params,
                                cv=self.cv,
                                seed=self.seed,
                                scoring=self._SCORE_RENAME_DICT[self.scoring],
                                mlflow_logging=mlflow_logging,
                                n_trials=n_iter,
                                **tuning_kws
                                )
        else:
            raise Exception('`tuning_algo` should be "grid", "random", "bayes-opt", "optuna"')

    def _flow_and_run_tuning(self, tuner, estimator, tuning_params, n_iter, learner_name, tuning_kws):
        """
        MLflowのセッティングとチューニング実行
        """
        # MLflow実行時
        if self.mlflow_logging:
            if self.mlflow_experiment_name is not None:
                experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                experiment_id = experiment.experiment_id
            else:
                experiment_id = None
            with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=learner_name) as run:
                self._run_tuning(tuner, estimator, tuning_params, n_iter, tuning_kws, mlflow_logging='outside')
        # MLflow実行しないとき
        else:
            self._run_tuning(tuner, estimator, tuning_params, n_iter, tuning_kws, mlflow_logging=None)
    
    def _score_correction(self, score_src, score_name):
        """
        評価指標をチューニング用 → 本来の定義の値に補正
        """
        if score_name == 'logloss':
            scores_fixed = -score_src
        elif score_name == 'rmse':
            scores_fixed = -score_src
        elif score_name == 'mse':
            scores_fixed = -score_src
        elif score_name == 'mae':
            scores_fixed = -score_src
        elif score_name == 'rmsle':
            scores_fixed = np.sqrt(-score_src)
        elif score_name == 'mape':
            scores_fixed = -score_src
        else:
            scores_fixed = score_src
        return scores_fixed

    def _retain_tuning_result(self, tuner, learner_name):
        """
        チューニング結果の保持
        """
        # 分類タスクかつXGBoostのときのみ、目的変数をint型に変換
        if learner_name == 'xgboost' and self.objective in ['binary', 'multiclass']:
            le = LabelEncoder()
            le.fit(self.y)
            y_trans = le.transform(self.y)
        else:
            y_trans = self.y
        # チューニング結果を保持
        self.best_scores[learner_name] = self._score_correction(tuner.best_score, self.scoring)  # スコアの保持
        self.tuners[learner_name] = tuner  # チューニング用インスタンスの保持
        # チューニング前の学習器を保持
        self.estimators_before[learner_name] = copy.deepcopy(tuner.estimator)
        params_before = {}
        params_before.update(tuner.not_opt_params)
        self.estimators_before[learner_name].set_params(**params_before)
        self.estimators_before[learner_name].fit(self.X, y_trans, **tuner.fit_params)
        # チューニング後の学習器を保持
        self.estimators_after[learner_name] = copy.deepcopy(tuner.estimator)
        params_after = {}
        params_after.update(tuner.not_opt_params)
        params_after.update(tuner.best_params)
        self.estimators_after[learner_name].set_params(**params_after)
        self.estimators_after[learner_name].fit(self.X, y_trans, **tuner.fit_params)

    def _regression_tuning(self, learner_name):
        """
        回帰のチューニング実行
        """
        print(f'Start {learner_name} tuning')
        # チューニングに使用する引数をプロパティから取得
        estimator = self.estimators[learner_name]
        tuning_params = self.tuning_params[learner_name]
        tuning_kws = self.tuning_kws[learner_name]
        n_iter = self.n_iter[learner_name] if learner_name in self.n_iter.keys() else None
        # 線形回帰 (チューニングなし)
        if learner_name == 'linear_regression':
            tuner = LinearRegressionTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # ElasticNet
        elif learner_name == 'elasticnet':
            tuner = ElasticNetTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # サポートベクター回帰
        elif learner_name == 'svr':
            tuner = SVMRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # ランダムフォレスト回帰
        elif learner_name == 'randomforest':
            tuner = RFRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # LightGBM回帰
        elif learner_name == 'lightgbm':
            tuner = LGBMRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # XGBoost回帰
        elif learner_name == 'xgboost':
            tuner = XGBRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # チューニング実行
        self._flow_and_run_tuning(tuner, estimator, tuning_params, n_iter, learner_name, tuning_kws)
        
        # チューニング結果の保持
        self._retain_tuning_result(tuner, learner_name)

    def _classification_tuning(self, learner_name):
        """
        分類のチューニング実行
        """
        print(f'Start {learner_name} tuning')
        # チューニングに使用する引数をプロパティから取得
        estimator = self.estimators[learner_name]
        tuning_params = self.tuning_params[learner_name]
        tuning_kws = self.tuning_kws[learner_name]
        n_iter = self.n_iter[learner_name] if learner_name in self.n_iter.keys() else None

        # サポートベクターマシン
        if learner_name == 'svm':
            tuner = SVMClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # ロジスティック回帰
        elif learner_name == 'logistic':
            tuner = LogisticRegressionTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # ランダムフォレスト
        elif learner_name == 'randomforest':
            tuner = RFClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # LightGBM分類
        elif learner_name == 'lightgbm':
            tuner = LGBMClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # XGBoost分類
        elif learner_name == 'xgboost':
            tuner = XGBClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
        # チューニング実行
        self._flow_and_run_tuning(tuner, estimator, tuning_params, n_iter, learner_name, tuning_kws)

        # チューニング結果の保持
        self._retain_tuning_result(tuner, learner_name)

    def _plot_regression_pred_true(self, learner_name, ax, after_tuning):
        """
        回帰モデルの予測値vs実測値プロット
        """
        tuner = self.tuners[learner_name]
        estimator = copy.deepcopy(tuner.estimator)
        params = {}
        params.update(tuner.not_opt_params)
        if after_tuning:  # チューニング後のモデルを表示したいとき
            params.update(tuner.best_params)
        estimator.set_params(**params)
        regplot.regression_pred_true(estimator, tuner.X, tuner.y, x_colnames=tuner.x_colnames,
                                     scores=self.scoring,
                                     cv=tuner.cv, ax=ax,
                                     fit_params=tuner.fit_params,
                                     legend_kws={'loc':'upper left'})
        # 一番上の行に学習器名を追加
        title_before = ax[0].title._text
        ax[0].set_title(f'{learner_name.upper()}\n\n{title_before}')

    def _plot_roc_curve(self, learner_name, ax, after_tuning):
        """
        分類モデルのROC曲線プロット
        """
        tuner = self.tuners[learner_name]
        estimator = copy.deepcopy(tuner.estimator)
        fit_params = copy.deepcopy(tuner.fit_params)
        params = {}
        params.update(tuner.not_opt_params)
        # LightGBM、XGBoostでobjectiveにmulticlassを指定しているとき、OVRでエラーが出るのでobjectiveを削除
        if 'objective' in params and params['objective'] in ['multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr', 'multi:softmax', 'multi:softprob']:
            print(f'The `objective` argument of {learner_name} set from "{params["objective"]}" to None because multiclass objective is not available in One-Vs-Rest Classifier')
            params.pop('objective')
        # LightGBM、XGBoostでeval_metricにmulticlassを指定しているとき、OVRでエラーが出るので置換
        if 'eval_metric' in fit_params:
            fit_params['eval_metric'] = fit_params['eval_metric'].replace(
                                'multi_logloss', 'binary_logloss').replace(
                                'multi_error', 'binary_error').replace(
                                'mlogloss', 'logloss')
        # チューニング後のモデルを表示したいとき
        if after_tuning:
            params.update(tuner.best_params)
        estimator.set_params(**params)
        classplot.roc_plot(estimator, tuner.X, tuner.y, x_colnames=tuner.x_colnames,
                           cv=tuner.cv, ax=ax,
                           fit_params=fit_params)
        # 一番上の行に学習器名を追加
        title_before = ax[0].title._text
        ax[0].set_title(f'{learner_name.upper()}\n\n{title_before}')

    def _calc_all_scores(self, learner_name, after_tuning):
        """
        チューニング対象以外のスコアを算出
        """
        tuner = self.tuners[learner_name]
        estimator = copy.deepcopy(tuner.estimator)
        # パラメータをセット
        params = {}
        params.update(tuner.not_opt_params)
        if after_tuning:  # チューニング後のモデルから取得したいとき
            params.update(tuner.best_params)
        estimator.set_params(**params)
        # 分類タスクのとき、ラベルをint型に変換 (strだとprecision, recall, f1が算出できない)
        fit_params = copy.deepcopy(tuner.fit_params)
        if self.objective in ['binary', 'multiclass']:
            le = LabelEncoder()
            le.fit(tuner.y)
            y_trans = le.transform(tuner.y)
            # fit_paramsがstr型のとき、int型に変換
            if 'eval_set' in fit_params and fit_params['eval_set'][0][1].dtype.name == 'object':
                fit_params['eval_set'] = [(fit_params['eval_set'][0][0], le.transform(fit_params['eval_set'][0][1]))]
        else:
            y_trans = tuner.y
        # スコア算出
        scores = cross_validate(estimator, tuner.X, y_trans,
                                groups=tuner.cv_group,
                                scoring={k: self._SCORE_RENAME_DICT[k] for k in self.other_scores},
                                cv = tuner.cv,
                                fit_params=fit_params
                                )
        # スコア名の'test_'文字列を除外
        scores = {k.replace('test_', ''): v for k, v in scores.items() if k not in ['fit_time', 'score_time']}
        # スコアを定義通りに補正
        scores = {k: np.vectorize(lambda x: self._score_correction(x, k))(v) for k, v in scores.items()}
        # 平均値算出
        scores_mean = {k: np.nanmean(v) for k, v in scores.items()}
        # 学習器名とチューニング前後を記載
        scores['learning_algo'] = learner_name
        scores['after_tuning'] = after_tuning
        scores_mean['learning_algo'] = learner_name
        scores_mean['after_tuning'] = after_tuning

        return scores, scores_mean

    def _log_mlflow_results(self):
        # タグを記載
        mlflow.set_tag('tuning_algo', self.tuning_algo)  # 最適化に使用したアルゴリズム名('grid', 'random', 'bayes-opt', 'optuna')
        mlflow.set_tag('x_colnames', self.x_colnames)  # 説明変数のカラム名
        mlflow.set_tag('y_colname', self.y_colname)  # 目的変数のカラム名
        # 引数をParametersとして保存
        mlflow.log_param('objective', self.objective)  # タスク ('regression', 'binary', 'multiclass')
        mlflow.log_param('scoring', self.scoring)  # チューニング非対象のパラメータ
        mlflow.log_param('other_scores', self.other_scores)  # チューニング後に表示するスコア一覧
        mlflow.log_param('learning_algos', self.learning_algos)  # 比較する学習器の一覧
        mlflow.log_param('n_iter', self.n_iter)  # 学習器ごとのチューニング試行数 (グリッドサーチ以外で有効)
        mlflow.log_param('cv', str(self.cv))  # クロスバリデーション分割法
        mlflow.log_param('seed', self.seed)  # 乱数シード
        # dict引数をJSONとして保存
        estimators = {k: str(v) for k, v in self.estimators.items()}
        mlflow.log_dict(estimators, 'arg-estimators.json')  # 最適化対象の学習器インスタンス
        mlflow.log_dict(self.tuning_params, 'arg-tuning_params.json')  # 学習時のパラメータ
        mlflow.log_dict(self.tuning_kws, 'arg-tuning_kws.json')  # 学習時のパラメータ
        # スコア履歴をMetricsとして保存
        for i, learner_name in enumerate(self.learning_algos):
            df_history = self.tuners[learner_name].get_search_history()
            for i, row in df_history.iterrows():
                mlflow.log_metric(f'score_history_{learner_name}', row['max_score'], step=i)

    def _tune_and_score(self, cv_num, n_learners):
        """チューニング実行とスコア比較"""
        ###### チューニング実行 ######
        for i, learner_name in enumerate(self.learning_algos):
            # 回帰のとき
            if self.objective == 'regression':
                self._regression_tuning(learner_name)
            # 分類のとき
            elif self.objective in ['binary', 'multiclass']:
                self._classification_tuning(learner_name)

        ###### スコア上昇履歴プロット ######
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle(f'{self.scoring} increase history')
        for i, learner_name in enumerate(self.learning_algos):
            self.tuners[learner_name].plot_search_history(ax=ax, x_axis='time',
                                            plot_kws={'color':self._COLOR_LIST[i],
                                                      'label':learner_name
                                                      })
        plt.legend()
        if self.mlflow_logging:  # MLflowに図を保存
            mlflow.log_figure(fig, 'score_history.png')
        plt.show()

        ###### 回帰のとき、チューニング前後の予測値vs実測値プロット ######
        if self.objective == 'regression':
            # チューニング前
            fig, axes = plt.subplots(cv_num + 1, n_learners, figsize=(n_learners*4, (cv_num+1)*4))
            fig.suptitle(f'Estimators BEFORE tuning', fontsize=18)
            ax_pred = [[row[i] for row in axes] for i in range(n_learners)] if n_learners > 1 else [axes]
            for i, learner_name in enumerate(self.learning_algos):
                self._plot_regression_pred_true(learner_name, ax_pred[i],
                                                after_tuning=False)
            fig.tight_layout(rect=[0, 0, 1, 0.98])  # https://tm23forest.com/contents/matplotlib-tightlayout-with-figure-suptitle
            if self.mlflow_logging:  # MLflowに図を保存
                mlflow.log_figure(fig, 'pred_true_before.png')
            plt.show()
            # チューニング後
            fig, axes = plt.subplots(cv_num + 1, n_learners, figsize=(n_learners*4, (cv_num+1)*4))
            fig.suptitle(f'Estimators AFTER tuning', fontsize=18)
            ax_pred = [[row[i] for row in axes] for i in range(n_learners)] if n_learners > 1 else [axes]
            for i, learner_name in enumerate(self.learning_algos):
                self._plot_regression_pred_true(learner_name, ax_pred[i],
                                                after_tuning=True)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            if self.mlflow_logging:  # MLflowに図を保存
                mlflow.log_figure(fig, 'pred_true_after.png')
            plt.show()

        ###### 分類のとき、ROC曲線をプロット ######
        else:
            # チューニング前
            fig, axes = plt.subplots(cv_num + 1, n_learners, figsize=(n_learners*6, (cv_num+1)*6))
            fig.suptitle(f'ROC curve BEFORE tuning', fontsize=18)
            ax_pred = [[row[i] for row in axes] for i in range(n_learners)] if n_learners > 1 else [axes]
            for i, learner_name in enumerate(self.learning_algos):
                self._plot_roc_curve(learner_name, ax_pred[i],
                                                after_tuning=False)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            if self.mlflow_logging:  # MLflowに図を保存
                mlflow.log_figure(fig, 'roc_curve_before.png')
            plt.show()
            # チューニング後
            fig, axes = plt.subplots(cv_num + 1, n_learners, figsize=(n_learners*6, (cv_num+1)*6))
            fig.suptitle(f'ROC curve AFTER tuning', fontsize=18)
            ax_pred = [[row[i] for row in axes] for i in range(n_learners)] if n_learners > 1 else [axes]
            for i, learner_name in enumerate(self.learning_algos):
                self._plot_roc_curve(learner_name, ax_pred[i],
                                                after_tuning=True)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            if self.mlflow_logging:  # MLflowに図を保存
                mlflow.log_figure(fig, 'roc_curve_after.png')
            plt.show()


        ###### チューニング対象以外のスコアを算出 ######
        scores_list = []
        scores_cv_list = []
        # チューニング前のスコア
        for learner_name in self.learning_algos:
            scores_cv, scores_mean = self._calc_all_scores(learner_name, False)
            scores_cv_list.append(scores_cv)
            scores_list.append(scores_mean)
        # チューニング後のスコア
        for learner_name in self.learning_algos:
            scores_cv, scores_mean = self._calc_all_scores(learner_name, True)
            scores_cv_list.append(scores_cv)
            scores_list.append(scores_mean)
        df_scores = pd.DataFrame(scores_list)
        df_scores_cv = pd.DataFrame(scores_cv_list)
        self.df_scores = df_scores
        self.df_scores_cv = df_scores_cv
        # MLflowにスコア一覧をCSVとして保存
        if self.mlflow_logging:
            df_scores.to_csv('score_result.csv')
            mlflow.log_artifact('score_result.csv')
            os.remove('score_result.csv')
            df_scores_cv.to_csv('score_result_cv.csv')
            mlflow.log_artifact('score_result_cv.csv')
            os.remove('score_result_cv.csv')

        ###### 最も性能の良い学習器の保持と表示 ######
        if self._SCORE_NEGATIVE[self.scoring]:  # 小さい方がGoodなスコアのとき
            best_idx = df_scores[df_scores['after_tuning']][self.scoring].idxmin()
        else:  # 大きい方がGoodなスコアのとき
            best_idx = df_scores[df_scores['after_tuning']][self.scoring].idxmax()
        self.best_learner = df_scores.loc[best_idx]['learning_algo']

        ###### print_estimator ######
        self.print_estimator(self.best_learner, mlflow_logging=self.mlflow_logging)

        ###### MLflowにパラメータと結果をロギング ######
        if self.mlflow_logging:
            self._log_mlflow_results()

        return df_scores

    def print_estimator(self, learner_name, mlflow_logging=False):
        """
        Print estimator after tuning

        Parameters
        ----------
        learner_name : {'linear_regression', 'elasticnet', 'svr', 'randomforest', 'lightgbm', 'xgboost', 'svm', 'logistic'}, or np.ndarray
            Printed learning algorithm name
        """

        print('----------The following is how to use the best estimator----------\n')
        tuner = self.tuners[learner_name]
        printed_model = []

        # importの表示
        if self.objective == 'regression':
            if learner_name == 'linear_regression':
                printed_model.append('from sklearn.linear_model import LinearRegression')
            elif learner_name == 'elasticnet':
                printed_model.append('from sklearn.linear_model import ElasticNet')
            elif learner_name == 'svr':
                printed_model.append('from sklearn.svm import SVR')
            elif learner_name == 'randomforest':
                printed_model.append('from sklearn.ensemble import RandomForestRegressor')
            elif learner_name == 'lightgbm':
                printed_model.append('from lightgbm import LGBMRegressor')
            elif learner_name == 'xgboost':
                printed_model.append('from xgboost import XGBRegressor')
        else:
            if learner_name == 'svm':
                printed_model.append('from sklearn.svm import SVC')
            elif learner_name == 'logistic':
                printed_model.append('from sklearn.linear_model import LogisticRegression')
            elif learner_name == 'randomforest':
                printed_model.append('from sklearn.ensemble import RandomForestClassifier')
            elif learner_name == 'lightgbm':
                printed_model.append('from lightgbm import LGBMClassifier')
            elif learner_name == 'xgboost':
                printed_model.append('from xgboost import XGBClassifier')
        # パイプラインならimport追加
        if isinstance(tuner.estimator, Pipeline):
            printed_model.append('from sklearn.pipeline import Pipeline')
            printed_model.append('from sklearn.preprocessing import StandardScaler')

        # 学習器情報の表示
        printed_model.append(f'NOT_OPT_PARAMS = {str(tuner.not_opt_params)}')  # チューニング対象外パラメータ
        printed_model.append(f'BEST_PARAMS = {str(tuner.best_params)}')  # チューニング対象パラメータ
        printed_model.append('params = {}')
        printed_model.append('params.update(NOT_OPT_PARAMS)')
        printed_model.append('params.update(BEST_PARAMS)')
        printed_model.append(f'estimator = {str(tuner.estimator)}')  # 学習器
        printed_model.append('estimator.set_params(**params)')  # 学習器にパラメータをセット
        if tuner.fit_params == {}:  # fit_paramsがないとき
            printed_model.append('estimator.fit(X, y)')
        else:  # fit_paramsがあるとき
            if 'eval_set' in tuner.fit_params.keys():  # fit_paramsにeval_setが含まれるとき、[(X, y)]に置換
                fit_params = copy.deepcopy(tuner.fit_params)
                fit_params['eval_set'] = 'dummy'
                str_fit_params = str(fit_params)
                str_fit_params = str_fit_params.replace("'dummy'", "[(X, y)]")
            else:
                str_fit_params = str(tuner.fit_params)
            printed_model.append(f'FIT_PARAMS = {str_fit_params}')
            printed_model.append('estimator.fit(X, y, FIT_PARAMS)')
        
        # 作成したモデル文字列をPrint
        printed_model = '\n'.join(printed_model)
        print(printed_model)
        # MLflowにモデル文字列を保存
        if mlflow_logging:
            mlflow.log_text(printed_model, 'how_to_use_best_estimator.py')

    def muscle_brain_tuning(self, x, y, data=None, x_colnames=None, cv_group=None,
                            objective=None, 
                            scoring=None, other_scores=None, learning_algos=None, n_iter=None,
                            cv=5, tuning_algo='optuna', seed=42,
                            estimators=None, tuning_params=None,
                            mlflow_logging=False, mlflow_tracking_uri=None, mlflow_artifact_location=None, mlflow_experiment_name=None,
                            tuning_kws=None):
        """
        Parameter tuning with multiple estimators. Easy to use even if your brain is made of muscle.

        Parameters
        ----------
        x : list[str], or numpy.ndarray
            Explanatory variables. Should be list[str] if ``data`` is pd.DataFrame.
            Should be numpy.ndarray if ``data`` is None.
        
        y : str or numpy.ndarray
            Target variable. Should be str if ``data`` is pd.DataFrame.
            Should be numpy.ndarray if ``data`` is None.
        
        data : pd.DataFrame, default=None
            Input data structure.
        
        x_colnames : list[str], default=None
            Names of explanatory variables. Available only if data is NOT pd.DataFrame.
        
        cv_group : str or numpy.ndarray, default=None
            Grouping variable that will be used for GroupKFold or LeaveOneGroupOut.
            Should be str if ``data`` is pd.DataFrame.
        
        objective : {'classification', 'regression'}, default=None
            Specify the learning task.
            If None, select task by target variable automatically.
        
        scoring : str, default=None
            Score name used to parameter tuning.

            - In regression:
                - 'rmse' : Root mean squared error
                - 'mse' : Mean squared error
                - 'mae' : Mean absolute error
                - 'rmsle' : Rot mean absolute logarithmic error
                - 'mape' : Mean absolute percentage error
                - 'r2' : R2 Score

            - In binary classification:
                - 'logloss' : Logarithmic Loss
                - 'accuracy' : Accuracy
                - 'precision' : Precision
                - 'recall' : Recall
                - 'f1' : F1 score
                - 'pr_auc' : PR-AUC
                - 'auc' : AUC

            - In multiclass classification:
                - 'logloss' : Logarithmic Loss
                - 'accuracy' : Accuracy
                - 'precision_macro' : Precision macro
                - 'recall_macro' : Recall macro
                - 'f1_micro' : F1 micro
                - 'f1_macro' : F1 macro
                - 'f1_weighted' : F1 weighted
                - 'auc_ovr' : One-vs-rest AUC
                - 'auc_ovo' : One-vs-one AUC
                - 'auc_ovr' : One-vs-rest AUC weighted
                - 'auc_ovo' : One-vs-one AUC weighted
            
            If None, the `SCORING` constant is used.

            See https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.SCORING
        
        other_scores : list[str], default=None
            Score names calculated after tuning. Available score names are written in the explatnation of `scoring` argument.

            If None, the `OTHER_SCORES` constant is used.

            See https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.OTHER_SCORES

            .. note::"rmsle" score may causes an error if predicted values or true values include negative value.
        
        learning_algos : list[str], default=None
            Estimator algorithm. Select the following algorithms and make a list of them.

            - In regression:
                - 'linear_regression' : LinearRegression
                - 'elasticnet' : ElasticNet
                - 'svr' : SVR
                - 'randomforest' : RandomForestRegressor
                - 'lightgbm' : LGBMRegressor
                - 'xgboost' : XGBRegressor

            - In regression:
                - 'svm' : SVC
                - 'logistic' : LogisticRegression
                - 'randomforest' : RandomForestClassifier
                - 'lightgbm' : LGBMClassifier
                - 'xgboost' : XGBClassifier

            If None, the `LEARNING_ALGOS` constant is used.

            See https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.LEARNING_ALGOS
        
        n_iter : dict[str, int], default=None
            Iteration number of parameter tuning. Keys should be members of ``learning_algos`` argument.
            Values should be iteration numbers.

            If None, the `N_ITER` constant is used.

            See https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.N_ITER
        
        cv : int, cross-validation generator, or an iterable, default=5
            Determines the cross-validation splitting strategy. If None, to use the default 5-fold cross validation. If int, to specify the number of folds in a KFold.
        
        tuning_algo : {'grid', 'random', 'bo', 'optuna'}, default='optuna'
            Tuning algorithm using following libraries. 'grid': sklearn.model_selection.GridSearchCV, 'random': sklearn.model_selection.RandomizedSearchCV, 'bo': BayesianOptimization, 'optuna': Optuna.
        
        seed : int, default=42
            Seed for random number generator of cross validation, estimators, and optuna.sampler.
        
        estimators : dict[str, estimator object implementing 'fit'], default=None
            Classification or regression estimators used to tuning.
            Keys should be members of ``learning_algos`` argument.
            Values are assumed to implement the scikit-learn estimator interface.

            If None, use default estimators of tuning instances

            See https://c60evaporator.github.io/muscle-tuning/each_estimators.html
        
        tuning_params : dict[str, dict[str, {list, tuple}]], default=None
            Values should be dictionary with parameters names as keys and 
            lists of parameter settings or parameter range to try as values. 
            Keys should be members of ``learning_algos`` argument. 

            If None, use default values of tuning instances

            See https://c60evaporator.github.io/muscle-tuning/each_estimators.html
        
        mlflow_logging : str, default=None
            Strategy to record the result by MLflow library.

            If True, nested runs are created.
            The parent run records conparison of all estimatiors such as max score history.
            The child runs are created in each tuning instances by setting ``mlflow_logging`` argument to "outside"

            If False, MLflow runs are not created.

        mlflow_tracking_uri : str, default=None
            Tracking uri for MLflow. This argument is passed to ``tracking_uri`` in ``mlflow.set_tracking_uri()``

            See https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri

        mlflow_artifact_location : str, default=None
            Artifact store for MLflow. This argument is passed to ``artifact_location`` in ``mlflow.create_experiment()``

            See https://mlflow.org/docs/latest/tracking.html#artifact-stores

        mlflow_experiment_name : str, default=None
            Experiment name for MLflow. This argument is passed to ``name`` in ``mlflow.create_experiment()``

            See https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment

        tuning_kws : dict[str, dict], default=None
            Additional parameters passed to tuning instances.
            Keys should be members of ``learning_algos`` argument.
            Values should be dict of parameters passed to tuning instances, e.g. {'not_opt_params': {''kernel': 'rbf'}}.

            See API Reference of tuning instances. 

        Returns
        ----------
        df_result : pd.DataFrame
            Validation scores of before and after tuning model.
        """
        ###### プロパティの初期化 ######
        # データの整形
        self._initialize(x, y, data, x_colnames, cv_group)
        # タスクの判定
        self._select_objective(objective)
        # 定数からプロパティのデフォルト値読込
        self._set_property_from_const(scoring, other_scores, learning_algos, n_iter)
        # 引数からプロパティ読込
        self._set_property_from_arguments(cv, tuning_algo, seed, mlflow_logging, mlflow_tracking_uri, mlflow_artifact_location, mlflow_experiment_name)
        # チューニング用クラスのデフォルト値から読み込むプロパティ
        self._set_property_from_algo(estimators, tuning_params, tuning_kws)
        # 学習器の数
        n_learners = len(self.learning_algos)

        ###### クロスバリデーション分割数を取得 ######
        if isinstance(self.cv, LeaveOneGroupOut):
            cv_num = len(set(self.data[self.group_name].values))
        else:
            cv_num = self.cv.n_splits

        ###### チューニングと評価を実行 ######
        # MLflowによるロギング実行時
        if mlflow_logging:
            if mlflow_tracking_uri is not None:  # tracking_uri
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            if mlflow_experiment_name is not None:  # experiment
                experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
                if experiment is None:  # 当該experiment存在しないとき、新たに作成
                    experiment_id = mlflow.create_experiment(
                                            name=mlflow_experiment_name,
                                            artifact_location=mlflow_artifact_location)
                else: # 当該experiment存在するとき、IDを取得
                    experiment_id = experiment.experiment_id
            else:
                experiment_id = None
            # ロギング実行
            with mlflow.start_run(experiment_id=experiment_id) as run:
                df_scores = self._tune_and_score(cv_num, n_learners)
        else:
            df_scores = self._tune_and_score(cv_num, n_learners)
        

        return df_scores