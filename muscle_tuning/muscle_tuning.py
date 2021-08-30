from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import numpy as np
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
                    'multiclass': ['accuracy', 'precision', 'recall', 'f1_macro', 'logloss', 'auc_ovr'],
                    'regression': ['rmse', 'mae', 'rmsle', 'mape', 'r2']
                    }
    LEARNING_ALGOS = {'regression': ['linear_regression', 'elasticnet', 'svr', 'randomforest', 'lightgbm'],
                      'binary': ['svm', 'logistic', 'randomforest', 'lightgbm'],
                      'multiclass': ['svm', 'logistic', 'randomforest', 'lightgbm']
                      }
    N_TRIALS = {'regression': {'svr': 500,
                               'elasticnet': 500,
                               'randomforest': 500, 
                               'lightgbm': 200,
                               'xgboost': 100
                               },
                'binary': {'svm': 500,
                           'logistic': 500,
                           'randomforest': 500, 
                           'lightgbm': 200,
                           'xgboost': 100
                           },
                'multiclass': {'svm': 500,
                               'logistic': 500,
                               'randomforest': 500, 
                               'lightgbm': 200,
                               'xgboost': 100
                               }
                }
    
    SCORE_RENAME_DICT = {'logloss': 'neg_log_loss',
                         'auc': 'roc_auc',
                         'auc_ovr': 'roc_auc_ovr',
                         'rmse': 'neg_root_mean_squared_error',
                         'mae': 'neg_mean_absolute_error',
                         'rmsle': 'neg_mean_squared_log_error',
                         'mape': 'neg_mean_absolute_percentage_error',
                         }

    def _reshape_input_data(self, x, y, data, x_colnames):
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
            self.data = data[x + [y]]
            self.x_colnames = x
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
            self.data = pd.DataFrame(np.column_stack((self.X, self.y)),
                                     columns=self.x_colnames + ['objective_variable'])
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
        self._reshape_input_data(x, y, data, x_colnames)
        self.cv_group = cv_group  # GroupKFold, LeaveOneGroupOut用のグルーピング対象データ
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

    def _run_tuning(self, tuning, estimator, tuning_params, n_trials, tuning_kws):
        """
        チューニング用メソッド実行
        """
        # グリッドサーチ
        if self.tuning_algo == 'grid':
            tuning.grid_search_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self.SCORE_RENAME_DICT[self.scoring],
                                    **tuning_kws
                                    )
        # ランダムサーチ
        elif self.tuning_algo == 'random':
            tuning.grid_search_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self.SCORE_RENAME_DICT[self.scoring],
                                    n_iter=n_trials,
                                    **tuning_kws
                                    )
        # BayesianOptimization
        elif self.tuning_algo == 'bayes-opt':
            tuning.bayes_opt_tuning(estimator=estimator,
                                    tuning_params=tuning_params,
                                    cv=self.cv,
                                    seed=self.seed,
                                    scoring=self.SCORE_RENAME_DICT[self.scoring],
                                    n_iter=n_trials,
                                    **tuning_kws
                                    )
        # Optuna
        elif self.tuning_algo == 'optuna':
            tuning.optuna_tuning(estimator=estimator,
                                tuning_params=tuning_params,
                                cv=self.cv,
                                seed=self.seed,
                                scoring=self.SCORE_RENAME_DICT[self.scoring],
                                n_trials=n_trials,
                                **tuning_kws
                                )
        else:
            raise Exception('`tuning_algo` should be "grid", "random", "bayes-opt", "optuna"')
    
    def _score_correction(self, score_values, score_name):
        """
        評価指標をチューニング用 → 本来の定義の値に補正
        """
        if score_name == 'logloss':
            scores_fixed = np.vectorize(lambda x: -x)(score_values)
        elif score_name == 'rmse':
            scores_fixed = np.vectorize(lambda x: -x)(score_values)
        elif score_name == 'mae':
            scores_fixed = np.vectorize(lambda x: -x)(score_values)
        elif score_name == 'rmsle':
            scores_fixed = np.vectorize(lambda x: np.sqrt(-x))(score_values)
        elif score_name == 'mape':
            scores_fixed = np.vectorize(lambda x: -x)(score_values)
        else:
            scores_fixed = score_values
        return scores_fixed

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
            best_estimator = LinearRegression()
            best_estimator.fit(self.X, self.y)
            best_scores = cross_val_score(copy.deepcopy(best_estimator), self.X, self.y,
                                        groups=self.cv_group, scoring=self.SCORE_RENAME_DICT[self.scoring],
                                        cv=self.cv)
            print(f'Best {self.scoring} of {learner_name} = {np.mean(self._score_correction(best_scores, self.scoring))}')
            
        # ElasticNet
        elif learner_name == 'elasticnet':
            tuning = ElasticNetTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuning, estimator, tuning_params, n_trials, tuning_kws)
        # サポートベクター回帰
        elif learner_name == 'svr':
            tuning = SVMRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuning, estimator, tuning_params, n_trials, tuning_kws)
        # ランダムフォレスト回帰
        elif learner_name == 'randomforest':
            tuning = RFRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuning, estimator, tuning_params, n_trials, tuning_kws)
        # LightGBM回帰
        elif learner_name == 'lightgbm':
            tuning = LGBMRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuning, estimator, tuning_params, n_trials, tuning_kws)
        # XGBoost回帰
        elif learner_name == 'xgboost':
            tuning = XGBRegressorTuning(self.X, self.y, self.x_colnames, cv_group=self.cv_group)
            self._run_tuning(tuning, estimator, tuning_params, n_trials, tuning_kws)
        
        # スコアの保持
        self.best_scores[learner_name] = tuning.best_score
        # チューニング用インスタンスの保持
        self.tuners[learner_name] = tuning

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
        
        # チューニング実行
        for learner_name in self.learning_algos:
            # 回帰のとき
            if self.objective == 'regression':
                self._regression_tuning(learner_name)
            # 分類のとき