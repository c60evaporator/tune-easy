from muscle_tuning.linearregression_tuning import LinearRegressionTuning
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
from seaborn_analyzer import regplot, classplot
import numbers
import gc
import copy

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
                    'regression': ['rmse', 'mae', 'rmsle', 'mape', 'r2']
                    }
    LEARNING_ALGOS = {'regression': ['linear_regression', 'elasticnet', 'svr', 'randomforest', 'lightgbm'],
                      #'regression': ['linear_regression', 'elasticnet', 'svr'],
                      'binary': ['svm', 'logistic', 'randomforest', 'lightgbm'],
                      #'binary': ['svm'],
                      #'multiclass': ['svm', 'logistic', 'randomforest', 'lightgbm']
                      'multiclass': ['svm', 'logistic']
                      }
    N_TRIALS = {'regression': {'svr': 500,
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
                'multiclass': {'svm': 500,
                               'logistic': 500,
                               'randomforest': 300, 
                               'lightgbm': 200,
                               'xgboost': 100
                               }
                }
    
    _SCORE_RENAME_DICT = {'rmse': 'neg_root_mean_squared_error',
                          'mae': 'neg_mean_absolute_error',
                          'rmsle': 'neg_mean_squared_log_error',
                          'mape': 'neg_mean_absolute_percentage_error',
                          'r2': 'r2',
                          'logloss': 'neg_log_loss',
                          'auc': 'roc_auc',
                          'auc_ovr': 'roc_auc_ovr',
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
                       'mae': True,
                       'rmsle': True,
                       'mape': True,
                       'r2': False,
                       'logloss': True,
                       'auc': False,
                       'auc_ovr': False,
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
        入力ファイルの形式統一(pd.DataFrame or np.ndarray)
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
            self.X = x
            self.y = y.ravel()
            # x_colnameとXの整合性確認
            if x_colnames is None:
                self.x_colnames = range(x.shape(1))
            elif x.shape[1] != len(x_colnames):
                raise Exception('width of X must be equal to length of x_colnames')
            else:
                self.x_colnames = x_colnames
            self.y_colname = 'objective_variable'
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
        self.other_scores = None  # チューニング後に表示するスコアのdict
        self.learning_algos = None  # 学習器名称のdict
        self.n_trials = None  # 試行数のdict (グリッドサーチ以外で有効)
        # 引数指定したデフォルト値を読み込むプロパティ
        self.tuning_algo = None  # 最適化に使用したアルゴリズム名('grid', 'random', 'bayes-opt', 'optuna')
        self.cv = None  # クロスバリデーション用インスタンス
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
                raise Exception('Objective variable should be int or float if `objective` is "regression"')
        # 分類指定時
        elif objective == 'classification':
            y_distinct = len(np.unique(self.y))  # 目的変数の固有メンバー数
            if y_distinct <= 1:  # 固有メンバー数の下限は2
                raise Exception('Number of unique members of objective variable should be bigger than 1')
            elif y_distinct == 2:  # 2クラス分類
                self.objective = 'binary'
            elif y_distinct <= 20:  # 多クラス分類
                self.objective = 'multiclass'
            else:  # 固有メンバーは20が上限
                raise Exception('Number of unique members of objective variable should be less than 20')
        # タスク未指定時
        elif objective is None:
            # float型 or int型のとき、回帰タスクとみなす
            if objective_dtype in [np.int32, np.int64, np.float32, np.float64]:
                self.objective = 'regression'
            # str型 or bool型のとき、分類タスクとみなす
            elif objective_dtype in [np.object, np.bool]:
                y_distinct = len(np.unique(self.y))  # 目的変数の固有メンバー数
                if y_distinct <= 1:  # 固有メンバー数の下限は2
                    raise Exception('Number of unique members of objective variable should be bigger than 1')
                elif y_distinct == 2:  # 2クラス分類
                    self.objective = 'binary'
                elif y_distinct <= 20:  # 多クラス分類
                    self.objective = 'multiclass'
                else:  # 固有メンバーは20が上限
                    raise Exception('Number of unique members of objective variable should be less than 20')
        else:
            raise Exception('`objective` argument should be "regression" or "classification"')

    def _set_property_from_const(self, scoring, other_scores, learning_algos, n_trials):
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
        if n_trials is None:
            self.n_trials = self.N_TRIALS[self.objective]
        else:
            self.n_trials = n_trials
        
    def _set_property_from_arguments(self, cv, tuning_algo, seed):
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

    def _run_tuning(self, tuner, estimator, tuning_params, n_trials, tuning_kws):
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
                                    **tuning_kws
                                    )
        # ランダムサーチ
        elif self.tuning_algo == 'random':
            tuner.grid_search_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self._SCORE_RENAME_DICT[self.scoring],
                                    n_iter=n_trials,
                                    **tuning_kws
                                    )
        # BayesianOptimization
        elif self.tuning_algo == 'bayes-opt':
            tuner.bayes_opt_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self._SCORE_RENAME_DICT[self.scoring],
                                    n_iter=n_trials,
                                    **tuning_kws
                                    )
        # Optuna
        elif self.tuning_algo == 'optuna':
            tuner.optuna_tuning(estimator=estimator,
                                tuning_params=tuning_params,
                                cv=self.cv,
                                seed=self.seed,
                                scoring=self._SCORE_RENAME_DICT[self.scoring],
                                n_trials=n_trials,
                                **tuning_kws
                                )
        else:
            raise Exception('`tuning_algo` should be "grid", "random", "bayes-opt", "optuna"')
    
    def _score_correction(self, score_src, score_name):
        """
        評価指標をチューニング用 → 本来の定義の値に補正
        """
        if score_name == 'logloss':
            scores_fixed = -score_src
        elif score_name == 'rmse':
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
        # チューニング結果を保持
        self.best_scores[learner_name] = self._score_correction(tuner.best_score, self.scoring)  # スコアの保持
        self.tuners[learner_name] = tuner  # チューニング用インスタンスの保持
        # チューニング前の学習器を保持
        self.estimators_before[learner_name] = copy.deepcopy(tuner.estimator)
        params_before = {}
        params_before.update(tuner.not_opt_params)
        self.estimators_before[learner_name].set_params(**params_before)
        self.estimators_before[learner_name].fit(self.X, self.y, **tuner.fit_params)
        # チューニング後の学習器を保持
        self.estimators_after[learner_name] = copy.deepcopy(tuner.estimator)
        params_after = {}
        params_after.update(tuner.not_opt_params)
        params_after.update(tuner.best_params)
        self.estimators_after[learner_name].set_params(**params_after)
        self.estimators_after[learner_name].fit(self.X, self.y, **tuner.fit_params)

    def _regression_tuning(self, learner_name):
        """
        回帰のチューニング実行
        """
        print(f'Start {learner_name} tuning')
        # チューニングに使用する引数をプロパティから取得
        estimator = self.estimators[learner_name]
        tuning_params = self.tuning_params[learner_name]
        tuning_kws = self.tuning_kws[learner_name]
        n_trials = self.n_trials[learner_name] if learner_name in self.n_trials.keys() else None
        # 線形回帰 (チューニングなし)
        if learner_name == 'linear_regression':
            tuner = LinearRegressionTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)  
        # ElasticNet
        elif learner_name == 'elasticnet':
            tuner = ElasticNetTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # サポートベクター回帰
        elif learner_name == 'svr':
            tuner = SVMRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # ランダムフォレスト回帰
        elif learner_name == 'randomforest':
            tuner = RFRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # LightGBM回帰
        elif learner_name == 'lightgbm':
            tuner = LGBMRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # XGBoost回帰
        elif learner_name == 'xgboost':
            tuner = XGBRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        
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
        n_trials = self.n_trials[learner_name] if learner_name in self.n_trials.keys() else None

        # サポートベクターマシン
        if learner_name == 'svm':
            tuner = SVMClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)  
        # ロジスティック回帰
        elif learner_name == 'logistic':
            tuner = LogisticRegressionTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # ランダムフォレスト
        elif learner_name == 'randomforest':
            tuner = RFClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # LightGBM分類
        elif learner_name == 'lightgbm':
            tuner = LGBMClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)
        # XGBoost分類
        elif learner_name == 'xgboost':
            tuner = XGBClassifierTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuner, estimator, tuning_params, n_trials, tuning_kws)

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
        regplot.regression_pred_true(estimator, self.x_colnames, self.y_colname, self.data,
                                     scores=self.scoring,
                                     cv=self.cv, ax=ax,
                                     fit_params=tuner.fit_params,
                                     legend_kws={'loc':'upper left'})
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
            le.fit(self.y)
            y_trans = le.transform(self.y)
            if 'eval_set' in tuner.fit_params:
                fit_params['eval_set'] = [(fit_params['eval_set'][0][0], le.transform(fit_params['eval_set'][0][1]))]
        else:
            y_trans = self.y
        # スコア算出
        scores = cross_validate(estimator, self.X, y_trans,
                                groups=self.cv_group,
                                scoring={k: self._SCORE_RENAME_DICT[k] for k in self.other_scores},
                                cv = self.cv,
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

    def print_estimator(self, learner_name):
        """
        Print estimator after tuning

        Parameters
        ----------
        learner_name : {'linear_regression', 'elasticnet', 'svr', 'randomforest', 'lightgbm', 'xgboost', 'svm', 'logistic'}, or np.ndarray
            Printed learning algorithm name
        """

        print('-----------------The following is how to use best estimator-------------------\n')
        tuner = self.tuners[learner_name]

        # importの表示
        if self.objective == 'regression':
            if learner_name == 'linear_regression':
                print('from sklearn.linear_model import LinearRegression')
            elif learner_name == 'elasticnet':
                print('from sklearn.linear_model import ElasticNet')
            elif learner_name == 'svr':
                print('from sklearn.svm import SVR')
            elif learner_name == 'randomforest':
                print('from sklearn.ensemble import RandomForestRegressor')
            elif learner_name == 'lightgbm':
                print('from lightgbm import LGBMRegressor')
            elif learner_name == 'xgboost':
                print('from xgboost import XGBRegressor')
        else:
            if learner_name == 'svm':
                print('from sklearn.svm import SVC')
            elif learner_name == 'logistic':
                print('from sklearn.linear_model import LogisticRegression')
            elif learner_name == 'randomforest':
                print('from sklearn.ensemble import RandomForestClassifier')
            elif learner_name == 'lightgbm':
                print('from lightgbm import LGBMClassifier')
            elif learner_name == 'xgboost':
                print('from xgboost import XGBClassifier')
        # パイプラインならimport追加
        if isinstance(tuner.estimator, Pipeline):
            print('from sklearn.pipeline import Pipeline')
            print('from sklearn.preprocessing import StandardScaler')

        # 学習器情報の表示
        print(f'NOT_OPT_PARAMS = {str(tuner.not_opt_params)}')  # チューニング対象外パラメータ
        print(f'BEST_PARAMS = {str(tuner.best_params)}')  # チューニング対象パラメータ
        print('params = {}')
        print('params.update(NOT_OPT_PARAMS)')
        print('params.update(BEST_PARAMS)')
        print(f'estimator = {str(tuner.estimator)}')  # 学習器
        print('estimator.set_params(**params)')  # 学習器にパラメータをセット
        if tuner.fit_params == {}:  # fit_paramsがないとき
            print('estimator.fit(X, y)')
        else:  # fit_paramsがあるとき
            print(f'FIT_PARAMS = {str(tuner.fit_params)}')
            print('estimator.fit(X, y, FIT_PARAMS)')

    def muscle_brain_tuning(self, x, y, data=None, x_colnames=None, cv_group=None,
                            objective=None, 
                            scoring=None, other_scores=None, learning_algos=None, n_trials=None,
                            cv=5, tuning_algo='optuna', seed=42,
                            estimators=None, tuning_params=None,
                            tuning_kws=None):
        """
        Parameter tuning with multiple estimators. Easy to use even if your brain is made of muscle.

        Parameters
        ----------
        x : list[str], or np.ndarray
            Explanatory variables. Should be list[str] if ``data`` is pd.DataFrame.
        y : str or np.ndarray
            Objective variable. Should be str if ``data`` is pd.DataFrame.
        data : pd.DataFrame, optional
            Input data structure.
        x_colnames : list[str]
            Names of explanatory variables. Available only if data is NOT pd.DataFrame
        cv_group : str or np.ndarray
            Grouping variable that will be used for GroupKFold or LeaveOneGroupOut. Should be str if ``data`` is pd.DataFrame
        objective : {'classification', 'regression'}
            Specify the learning task. If None, select task by objective variable automatically.
        scoring : str, optional
            Score name used to parameter tuning, e.g. rmse, mae, logloss, accuracy, auc.
        other_scores : list[str], optional
            Score names calculated after tuning, e.g. rmse, mae, logloss, accuracy, auc.
        learning_algos : list[str], optional
            Estimator algorithm. 'svm': Support vector machine, 'svr': Support vector regression, 'logistic': Logistic Regression, 'elasiticnet': ElasticNet, 'randomforest': RandomForest, 'lightgbm': LightGBM, 'xgboost': XGBoost.
        n_trials : dict[str, int], optional
            Iteration number of parameter tuning. Keys should be members of ``algo`` argument. Values should be iteration numbers.
        cv : int, cross-validation generator, or an iterable, optional
            Determines the cross-validation splitting strategy. If None, to use the default 5-fold cross validation. If int, to specify the number of folds in a KFold.
        tuning_algo : {'grid', 'random', 'bo', 'optuna'}, optional
            Tuning algorithm using following libraries. 'grid': sklearn.model_selection.GridSearchCV, 'random': sklearn.model_selection.RandomizedSearchCV, 'bo': BayesianOptimization, 'optuna': Optuna.
        seed : int, optional
            Seed for random number generator of cross validation, estimators, and optuna.sampler
        estimators : dict[str, estimator object implementing 'fit'], optional
            Classification or regression estimators used to tuning. Keys should be members of ``algo`` argument. Values are assumed to implement the scikit-learn estimator interface.
        tuning_params : dict[str, dict[str, {list, tuple}]], optional
            Values should be dictionary with parameters names as keys and lists of parameter settings or parameter range to try as values. Keys should be members of ``algo`` argument. If None, use default values of tuning instances
        tuning_kws : dict[str, dict]
            Additional parameters passed to tuning instances. Keys should be members of ``algo`` argument. Values should be dict of parameters passed to tuning instances, e.g. {'not_opt_params': {''kernel': 'rbf'}}

        Returns
        ----------
        df_result : pd.DataFrame
            Validation scores, e.g. r2, mae and rmse
        """
        ###### プロパティの初期化 ######
        # データの整形
        self._initialize(x, y, data, x_colnames, cv_group)
        # タスクの判定
        self._select_objective(objective)
        # 定数からプロパティのデフォルト値読込
        self._set_property_from_const(scoring, other_scores, learning_algos, n_trials)
        # 引数からプロパティ読込
        self._set_property_from_arguments(cv, tuning_algo, seed)
        # チューニング用クラスのデフォルト値から読み込むプロパティ
        self._set_property_from_algo(estimators, tuning_params, tuning_kws)
        # 学習器の数
        n_learners = len(self.learning_algos)

        # クロスバリデーション分割数を取得
        if isinstance(self.cv, LeaveOneGroupOut):
            cv_num = len(set(self.data[self.group_name].values))
        else:
            cv_num = self.cv.n_splits

        # チューニング実行
        for i, learner_name in enumerate(self.learning_algos):
            # 回帰のとき
            if self.objective == 'regression':
                self._regression_tuning(learner_name)
            # 分類のとき
            elif self.objective in ['binary', 'multiclass']:
                self._classification_tuning(learner_name)

        # スコア上昇履歴プロット
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle(f'{self.scoring} increase history')
        for i, learner_name in enumerate(self.learning_algos):
            self.tuners[learner_name].plot_search_history(ax=ax, x_axis='time',
                                            plot_kws={'color':self._COLOR_LIST[i],
                                                      'label':learner_name
                                                      })
        plt.legend()
        plt.show()

        # 回帰のとき、チューニング前後の予測値vs実測値プロット
        if self.objective == 'regression':
            # チューニング前
            fig, axes = plt.subplots(cv_num + 1, n_learners, figsize=(n_learners*4, (cv_num+1)*4))
            fig.suptitle(f'Estimators BEFORE tuning', fontsize=18)
            ax_pred = [[row[i] for row in axes] for i in range(n_learners)]
            for i, learner_name in enumerate(self.learning_algos):
                self._plot_regression_pred_true(learner_name, ax_pred[i],
                                                after_tuning=False)
            fig.tight_layout(rect=[0, 0, 1, 0.98])  # https://tm23forest.com/contents/matplotlib-tightlayout-with-figure-suptitle
            plt.show()
            # チューニング後
            fig, axes = plt.subplots(cv_num + 1, n_learners, figsize=(n_learners*4, (cv_num+1)*4))
            fig.suptitle(f'Estimators AFTER tuning', fontsize=18)
            ax_pred = [[row[i] for row in axes] for i in range(n_learners)]
            for i, learner_name in enumerate(self.learning_algos):
                self._plot_regression_pred_true(learner_name, ax_pred[i],
                                                after_tuning=True)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            plt.show()

        # 分類のとき、ROC曲線をプロット


        # チューニング対象以外のスコアを算出
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

        # 最も性能の良い学習器の保持
        if self._SCORE_NEGATIVE[self.scoring]:  # 小さい方がGoodなスコアのとき
            best_idx = df_scores[df_scores['after_tuning']][self.scoring].idxmin()
        else:  # 大きい方がGoodなスコアのとき
            best_idx = df_scores[df_scores['after_tuning']][self.scoring].idxmax()
        self.best_learner = df_scores.loc[best_idx]['learning_algo']
        # print_estimator
        self.print_estimator(self.best_learner)

        return df_scores