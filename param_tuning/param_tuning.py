from abc import abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, GroupKFold, LeaveOneGroupOut, validation_curve, learning_curve, cross_val_score
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
import optuna
import time
import numbers
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os

from .util_methods import round_digits

class ParamTuning():
    """
    パラメータチューニング用基底クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス
    ESTIMATOR = None
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
     # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = None

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {}

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {}

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの繰り返し回数
    CV_PARAMS_RANDOM = {}

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 120  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 10  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {}
    INT_PARAMS = []  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # Optuna用パラメータ
    N_ITER_OPTUNA = 300

    # 検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {}  # パラメータ範囲
    PARAM_SCALES = {}  # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')

    def _additional_init(self, **kwargs):
        """
        初期化時の追加処理 (継承先で記述)
        """
        pass

    def __init__(self, X, y, x_colnames, y_colname=None, cv_group=None, **kwargs):
        """
        初期化

        Parameters
        ----------
        X : np.ndarray
            説明変数データ(pandasではなく2次元ndarray)
        y : np.ndarray
            目的変数データ(ndarray、2次元でも1次元でも可)
        x_colnames : list(str)
            説明変数のフィールド名
        y_colname : str, optional
            目的変数のフィールド名
        cv_group: str, optional
            GroupKFold、LeaveOneGroupOutのグルーピング対象データ
        eval_data_source: {'all', 'valid', 'train'}
            self.eval_dataの指定方法 (XGBoost、LightGBMのみ有効)
        """
        if X.shape[1] != len(x_colnames):
            raise Exception('width of X must be equal to length of x_colnames')
        self.X = X
        self.y = y.ravel() # 2次元ndarrayのとき、ravel()で1次元に変換
        self.x_colnames = x_colnames
        self.y_colname = y_colname
        self.cv_group = cv_group  # GroupKFold, LeaveOneGroupOut用のグルーピング対象データ 
        self.tuning_params = None  # チューニング対象のパラメータとその範囲
        self.not_opt_params = None  # チューニング非対象のパラメータ
        self.int_params = None  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
        self.param_scales = None  # パラメータのスケール('linear', 'log')
        self.scoring = None  # 最大化するスコア
        self.seed = None  # 乱数シード
        self.cv = None  # クロスバリデーション分割法
        self.estimator = None  # 最適化対象の学習器インスタンス
        self.learner_name = None  # パイプライン処理時の学習器名称
        self.fit_params = None  # 学習時のパラメータ
        self.algo_name = None  # 最適化に使用したアルゴリズム名('grid', 'random', 'bayes-opt', 'optuna')
        self.best_params = None  # 最適パラメータ
        self.best_score = None  # 最高スコア
        self.elapsed_time = None  # 所要時間
        self.best_estimator = None  # 最適化された学習モデル
        self.search_history = None  # 探索履歴(パラメータ名をキーとしたdict)
        self.param_importances = None  # ランダムフォレストで求めたパラメータのスコアに対する重要度
        self.start_time = None  # 処理時間計測用スタート値
        self.preprocess_time = None  # 前処理(最適化スタート前)時間
        self.elapsed_times = None  # 処理経過時間保存用リスト
        # 追加処理
        self._additional_init(**kwargs)
    
    def _train_param_generation(self, src_fit_params):
        """
        入力データから学習時パラメータの生成 (例: XGBoostのeval_list)
        通常はデフォルトのままだが、必要であれば継承先でオーバーライド

        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """
        return src_fit_params
    
    def _tuning_param_generation(self, src_params):
        """
        入力データからチューニング用パラメータの生成 (例: ランダムフォレストのmax_features)
        通常はデフォルトのままだが、必要であれば継承先でオーバーライド

        Parameters
        ----------
        src_params : Dict
            処理前のチューニング用パラメータ
        """
        return src_params
    
    def _not_opt_param_generation(self, src_not_opt_params, seed, scoring):
        """
        チューニング対象外パラメータの生成(例: seed追加、評価指標の不整合チェック、loglossかつSVRのときのprobablity設定など)
        通常はrandom_state追加のみだが、必要であれば継承先でオーバーライド

        Parameters
        ----------
        src_not_opt_params : Dict
            処理前のチューニング対象外パラメータ
        seed : int
            乱数シード
        scoring : str
            最適化で最大化する評価指標
        
        """
        # 乱数シードをnot_opt_paramsのrandom_state引数に追加
        if 'random_state' in src_not_opt_params:
            src_not_opt_params['random_state'] = seed
        return src_not_opt_params
    
    def _set_argument_to_property(self, estimator, tuning_params, cv, seed, scoring, fit_params, not_opt_params, param_scales):
        """
        引数をプロパティ(インスタンス変数)に反映
        """
        self.estimator = estimator
        self.tuning_params = tuning_params
        self.cv = cv
        self.seed = seed
        self.scoring = scoring
        self.fit_params = fit_params
        self.not_opt_params = not_opt_params
        self.param_scales = param_scales

    def _add_learner_name(self, estimator, params):
        """
        パイプライン処理用に、パラメータ名を"学習器名__パラメータ名"に変更
        """
        if isinstance(estimator, Pipeline):
            # 学習器名が指定されているとき、パラメータ名を変更して処理を進める(既にパラメータ名に'__'が含まれているパラメータは、変更しない)
            if self.learner_name is not None:
                if isinstance(params, dict):  # Dictのとき
                    params = {k if '__' in k else f'{self.learner_name}__{k}': v for k, v in params.items()}
                elif isinstance(params, list):  # Listのとき
                    params = [param if '__' in param else f'{self.learner_name}__{param}' for param in params]
            # 指定されていないとき、エラーを返す
            else:
                raise Exception('Pipeline needs "lerner_name" argument')
        return params
    
    def _get_final_estimator_name(self, estimator):
        """
        パイプライン処理のとき、最後の要素から学習器名を取得
        """
        if isinstance(estimator, Pipeline):
            steps = estimator.steps
            self.learner_name = steps[len(steps)-1][0]

    def _scratch_cross_val(self, estimator, eval_data_source):
        scores = []
        for train, test in self.cv.split(self.X, self.y):
            X_train = self.X[train]
            y_train = self.y[train]
            X_test = self.X[test]
            y_test = self.y[test]
            # fitメソッド実行時のパラメータ指定
            fit_params = self.fit_params
            fit_params['verbose'] = 0
            # eval_setにテストデータを使用
            if eval_data_source == 'valid':
                fit_params['eval_set'] = [(X_test, y_test)]
            # eval_setに学習データを使用
            elif eval_data_source == 'train':
                fit_params['eval_set'] = [(X_train, y_train)]
            else:
                raise Exception('the "eval_data_source" argument must be "all", "valid", or "train"')
            # 学習
            estimator.fit(X_train, y_train,
                            **fit_params)
            scorer = check_scoring(estimator, self.scoring)
            score = scorer(estimator, X_test, y_test)

            # Learning API -> Scikit-learn APIとデフォルトパラメータが異なり結果が変わるので不使用
            # dtrain = xgb.DMatrix(X_train, label=y_train)
            # dtest = xgb.DMatrix(X_test, label=y_test)
            # evals = [(dtrain, 'train'), (dtest, 'eval')]
            # d_fit_params = {k: v for k, v in fit_params.items()}
            # d_fit_params['num_boost_round'] = 1000
            # d_fit_params.pop('eval_set')
            # d_fit_params.pop('verbose')
            # dmodel = xgb.train(params, dtrain, evals=evals, **d_fit_params)
            # pred2 = dmodel.predict(dtest)
            
            scores.append(score)
        return scores

    def _mlflow_logging(self):
        """
        MLFlowで各種情報と探索履歴をロギング
        """
        # パラメータを記載
        mlflow.log_param('X_head', self.X[:5, :])  # 説明変数の最初5レコード
        mlflow.log_param('y_head', self.y[:5])  # 目的変数の最初5レコード
        mlflow.log_param('x_colnames', self.x_colnames)  # 説明変数のカラム名
        mlflow.log_param('y_colname', self.y_colname)  # 目的変数のカラム名
        mlflow.log_param('cv_group_head', self.cv_group[:5] if self.cv_group is not None else None)  # GroupKFold, LeaveOneGroupOut用のグルーピング対象データ 
        mlflow.log_param('tuning_params', self.tuning_params)  # チューニング対象のパラメータとその範囲
        mlflow.log_param('not_opt_params', self.not_opt_params)  # チューニング非対象のパラメータ
        mlflow.log_param('int_params', self.int_params)  # 整数型のパラメータのリスト
        mlflow.log_param('param_scales', self.int_params)  # パラメータのスケール('linear', 'log')
        mlflow.log_param('scoring', self.scoring)  # 最大化するスコア
        mlflow.log_param('seed', self.seed)  # 乱数シード
        mlflow.log_param('cv', str(self.cv))  # クロスバリデーション分割法
        mlflow.log_param('estimator', str(self.estimator))  # 最適化対象の学習器インスタンス
        mlflow.log_param('learner_name', self.learner_name)  # パイプライン処理時の学習器名称
        mlflow.log_param('fit_params', self.fit_params)  # 学習時のパラメータ
        mlflow.log_param('algo_name', self.algo_name)  # 最適化に使用したアルゴリズム名('grid', 'random', 'bayes-opt', 'optuna')
        # チューニング結果を記載
        best_params_float = {f'best__{k}':v for k, v in self.best_params.items() if isinstance(v, float) or isinstance(v, int)}
        mlflow.log_params(best_params_float)  # 最適パラメータ(数値型)
        best_params_str = {f'best__{k}':v for k, v in self.best_params.items() if not isinstance(v, float) and not isinstance(v, int)}
        mlflow.log_params(best_params_str)  # 最適パラメータ(数値型以外)
        mlflow.log_metric('best_score', self.best_score)  # 最高スコア
        mlflow.log_metric('elapsed_time', self.elapsed_time)  # 所要時間
        mlflow.log_metric('preprocess_time', self.preprocess_time)  # 前処理(最適化スタート前)時間
        # 最適モデルをMLFlow Modelsで保存(https://mlflow.org/docs/latest/models.html#how-to-log-models-with-signatures)
        estimator_output = self.best_estimator.predict(self.X)  # モデル出力
        estimator_output = pd.Series(estimator_output) if self.y_colname is None else pd.DataFrame(estimator_output, columns=[self.y_colname])
        signature = infer_signature(pd.DataFrame(self.X, columns=self.x_colnames),  # モデル入出力の型を自動判定
                                    estimator_output)
        mlflow.sklearn.log_model(self.best_estimator, f'best_estimator_{self.algo_name}', signature=signature)  # 最適化された学習モデル
        # パラメータと得点の履歴をCSV化してArtifactとして保存
        df_history = self.get_search_history()
        df_history.to_csv('search_history.csv')
        mlflow.log_artifact('search_history.csv')
        os.remove('search_history.csv')

    def grid_search_tuning(self, estimator=None, tuning_params=None, cv=None, seed=None, scoring=None,
                           not_opt_params=None, param_scales=None, mlflow_logging=None, grid_kws=None, fit_params=None):
        """
        グリッドサーチ＋クロスバリデーション

        Parameters
        ----------
        estimator : Dict
            最適化対象の学習器インスタンス。not_opt_paransで指定したパラメータは上書きされるので注意
        tuning_params : Dict[str, List[float]]
            最適化対象のパラメータ一覧
            Pipelineのときは{学習器名__パラメータ名:[パラメータの値候補],‥}で指定する必要あり
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        not_opt_params : Dict
            最適化対象外のパラメータ一覧
        param_scales : Dict
            パラメータのスケール('linear', 'log')(Noneならクラス変数PARAM_SCALESから取得)
        mlflow_logging : str
            MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)
        grid_kws : Dict
            sklearn.model_selection.GridSearchCVに渡す引数(estimator, param_grid, cv, scoring以外)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()
        self.start_time = start

        # 引数非指定時、クラス変数から取得
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        if tuning_params is None:
            tuning_params = self.CV_PARAMS_GRID
        if cv is None:
            cv = self.CV_NUM
        if seed is None:
            seed = self.SEED
        if scoring is None:
            scoring = self.SCORING
        if not_opt_params is None:
            not_opt_params = self.NOT_OPT_PARAMS
        if param_scales is None:
            param_scales = self.PARAM_SCALES
        if grid_kws is None:
            grid_kws = {}
        if fit_params is None:
            fit_params = self.FIT_PARAMS

        # 入力データからチューニング用パラメータの生成
        tuning_params = self._tuning_param_generation(tuning_params)
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # チューニング対象外パラメータの生成
        not_opt_params = self._not_opt_param_generation(not_opt_params, seed, scoring)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        tuning_params = self._add_learner_name(estimator, tuning_params)
        fit_params = self._add_learner_name(estimator, fit_params)
        param_scales = self._add_learner_name(estimator, param_scales)
        not_opt_params = self._add_learner_name(estimator, not_opt_params)
        
        # 引数をプロパティ(インスタンス変数)に反映（チューニング中にパラメータ入力されるため、モデルはdeepcopy）
        self._set_argument_to_property(copy.deepcopy(estimator), tuning_params, cv, seed, scoring, fit_params, not_opt_params, param_scales)
        # チューニング対象外パラメータをモデルに反映
        estimator.set_params(**not_opt_params)

        # グリッドサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        if 'n_jobs' not in grid_kws.keys():
            grid_kws['n_jobs'] = -1
        gridcv = GridSearchCV(estimator, tuning_params, cv=cv,
                              scoring=scoring, **grid_kws)

        # ここまでに掛かった前処理時間を測定
        self.preprocess_time = time.time() - start
        # グリッドサーチ実行（学習実行）
        gridcv.fit(self.X,
               self.y,
               groups=self.cv_group,
               **fit_params
               )
        self.elapsed_time = time.time() - start
        self.algo_name = 'grid'
        
        # 最適パラメータの表示と保持
        print('最適パラメータ ' + str(gridcv.best_params_))
        self.best_params = gridcv.best_params_
        self.best_score = gridcv.best_score_
        # 最適モデルの保持
        self.best_estimator = gridcv.best_estimator_
        # 学習履歴の保持
        self.search_history = {k: gridcv.cv_results_['param_' + k].data.astype(np.float64) for k, v in tuning_params.items() if len(v) >= 2}
        self.search_history['test_score'] = gridcv.cv_results_['mean_test_score']
        # 所要時間の保持
        self.search_history['fit_time'] = gridcv.cv_results_['mean_fit_time']
        self.search_history['score_time'] = gridcv.cv_results_['mean_score_time']
        cv_num = gridcv.n_splits_
        self.search_history['raw_trial_time'] = (self.search_history['fit_time'] + self.search_history['score_time']) * cv_num

        # MLFlowで記録
        if mlflow_logging == 'log':
            self._mlflow_logging()
        elif mlflow_logging == 'with':
            with mlflow.start_run() as run:
                self._mlflow_logging()
        elif mlflow_logging is not None:
            raise Exception('the "mlflow_logging" argument must be "log", "with" or None')

        # グリッドサーチで探索した最適パラメータ、最適スコアを返す
        return gridcv.best_params_, gridcv.best_score_

    def random_search_tuning(self, estimator=None, tuning_params=None, cv=None, seed=None, scoring=None,
                             n_iter=None,
                             not_opt_params=None, param_scales=None, mlflow_logging=None, rand_kws=None, fit_params=None):
        """
        ランダムサーチ＋クロスバリデーション

        Parameters
        ----------
        estimator : Dict
            最適化対象の学習器インスタンス。not_opt_paransで指定したパラメータは上書きされるので注意
        tuning_params : Dict[str, List[float]]
            最適化対象のパラメータ一覧
            Pipelineのときは{学習器名__パラメータ名:[パラメータの値候補],‥}で指定する必要あり
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        n_iter : int
            ランダムサーチの繰り返し回数
        not_opt_params : Dict
            最適化対象外のパラメータ一覧
        param_scales : Dict
            パラメータのスケール('linear', 'log')(Noneならクラス変数PARAM_SCALESから取得)
        mlflow_logging : str
            MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)
        rand_kws : Dict
            sklearn.model_selection.RondomizedSearchCVに渡す引数(estimator, param_grid, cv, scoring, n_iter以外)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()
        self.start_time = start

        # 引数非指定時、クラス変数から取得
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        if tuning_params is None:
            tuning_params = self.CV_PARAMS_RANDOM
        if cv is None:
            cv = self.CV_NUM
        if seed is None:
            seed = self.SEED
        if scoring is None:
            scoring = self.SCORING
        if n_iter is None:
            n_iter = self.N_ITER_RANDOM
        if not_opt_params is None:
            not_opt_params = self.NOT_OPT_PARAMS
        if param_scales is None:
            param_scales = self.PARAM_SCALES
        if rand_kws is None:
            rand_kws = {}
        if fit_params is None:
            fit_params = self.FIT_PARAMS
        
        # 入力データからチューニング用パラメータの生成
        tuning_params = self._tuning_param_generation(tuning_params)
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # チューニング対象外パラメータの生成
        not_opt_params = self._not_opt_param_generation(not_opt_params, seed, scoring)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        tuning_params = self._add_learner_name(estimator, tuning_params)
        fit_params = self._add_learner_name(estimator, fit_params)
        param_scales = self._add_learner_name(estimator, param_scales)
        not_opt_params = self._add_learner_name(estimator, not_opt_params)

        # 引数をプロパティ(インスタンス変数)に反映（チューニング中にパラメータ入力されるため、モデルはdeepcopy）
        self._set_argument_to_property(copy.deepcopy(estimator), tuning_params, cv, seed, scoring, fit_params, not_opt_params, param_scales)
        # チューニング対象外パラメータをモデルに反映
        estimator.set_params(**not_opt_params)

        # ランダムサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        if 'n_jobs' not in rand_kws.keys():
            rand_kws['n_jobs'] = -1
        if 'random_state' not in rand_kws.keys():
            rand_kws['random_state'] = seed
        randcv = RandomizedSearchCV(estimator, tuning_params, cv=cv, scoring=scoring,
                                    n_iter=n_iter, **rand_kws)
        
        # ここまでに掛かった前処理時間を測定
        self.preprocess_time = time.time() - start
        # ランダムサーチ実行
        randcv.fit(self.X,
               self.y,
               groups=self.cv_group,
               **fit_params
               )
        self.elapsed_time = time.time() - start
        self.algo_name = 'random'
        
        # 最適パラメータの表示と保持
        print('最適パラメータ ' + str(randcv.best_params_))
        self.best_params = randcv.best_params_
        self.best_score = randcv.best_score_
        # 最適モデルの保持
        self.best_estimator = randcv.best_estimator_
        # 学習履歴の保持
        self.search_history = {k: randcv.cv_results_['param_' + k].data.astype(np.float64) for k, v in tuning_params.items() if len(v) >= 2}
        self.search_history['test_score'] = randcv.cv_results_['mean_test_score']
        # 所要時間の保持
        self.search_history['fit_time'] = randcv.cv_results_['mean_fit_time']
        self.search_history['score_time'] = randcv.cv_results_['mean_score_time']
        cv_num = randcv.n_splits_
        self.search_history['raw_trial_time'] = (self.search_history['fit_time'] + self.search_history['score_time']) * cv_num

        # MLFlowで記録
        if mlflow_logging == 'log':
            self._mlflow_logging()
        elif mlflow_logging == 'with':
            with mlflow.start_run() as run:
                self._mlflow_logging()
        elif mlflow_logging is not None:
            raise Exception('the "mlflow_logging" argument must be "log", "with" or None')

        # ランダムサーチで探索した最適パラメータ、最適スコアを返す
        return randcv.best_params_, randcv.best_score_

    @abstractmethod
    def _bayes_evaluate(self, **kwargs):
        """
         ベイズ最適化時の評価指標算出メソッド
        """
        # 最適化対象のパラメータ
        params = kwargs
        params = self._pow10_conversion(params, self.param_scales)  # 対数パラメータは10のべき乗に変換
        params = self._int_conversion(params, self.int_params)  # 整数パラメータはint型に変換
        params.update(self.not_opt_params)  # 最適化対象以外のパラメータも追加
        # モデル作成
        estimator = self.estimator
        estimator.set_params(**params)

        # cross_val_scoreでクロスバリデーション
        scores = cross_val_score(estimator, self.X, self.y, cv=self.cv, groups=self.cv_group,
                                 scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        val = scores.mean()
        # 所要時間測定
        self.elapsed_times.append(time.time() - self.start_time)

        return val

    def _int_conversion(self, tuning_params, int_params):
        """
         ベイズ最適化パラメータのうち、整数のものを四捨五入してint型変換
        """
        tuning_params = {k: round(v) if k in int_params else v for k, v in tuning_params.items()}
        return tuning_params

    def _log10_conversion(self, tuning_params, param_scales):
        """
         ベイズ最適化パラメータのうち、対数スケールのものを対数変換（ベイズ最適化メソッドに渡す前に適用）
        """
        tuning_params_log = {k: (np.log10(v[0]), np.log10(v[1])) if param_scales[k] == 'log' else v for k, v in tuning_params.items()}
        return tuning_params_log

    def _pow10_conversion(self, params_log, param_scales):
        """
         ベイズ最適化パラメータのうち、対数スケールのものを10のべき乗変換（ベイズ最適化メソッド内で適用）
        """
        params = {k: np.power(10, v) if param_scales[k] == 'log' else v for k, v in params_log.items()}
        return params

    def bayes_opt_tuning(self, estimator=None, tuning_params=None, cv=None, seed=None, scoring=None,
                         n_iter=None, init_points=None, acq=None,
                         not_opt_params=None, int_params=None, param_scales=None, mlflow_logging=None, fit_params=None):
        """
        ベイズ最適化(BayesianOptimization)

        Parameters
        ----------
        estimator : Dict
            最適化対象の学習器インスタンス。not_opt_paransで指定したパラメータは上書きされるので注意
        tuning_params : Dict[str, Tuple(float, float)]
            最適化対象のパラメータ範囲　{パラメータ名:(パラメータの探索下限,上限),‥}で指定
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        n_iter : int
            ベイズ最適化の繰り返し回数
        init_points : int
            初期観測点の個数(ランダムな探索を何回行うか)
        acq : str
            獲得関数('ei', 'pi', 'ucb')
        not_opt_params : Dict
            最適化対象外のパラメータ一覧
        int_params : List
            整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
        param_scales : Dict
            パラメータ
            のスケール('linear', 'log')(Noneならクラス変数PARAM_SCALESから取得)
        mlflow_logging : str
            MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()
        self.start_time = start

        # 引数非指定時、クラス変数から取得
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        if tuning_params is None:
            tuning_params = self.BAYES_PARAMS
        if cv is None:
            cv = self.CV_NUM
        if seed is None:
            seed = self.SEED
        if scoring is None:
            scoring = self.SCORING
        if n_iter is None:
            n_iter = self.N_ITER_BAYES
        if init_points is None:
            init_points = self.INIT_POINTS
        if acq is None:
            acq = self.ACQ
        if not_opt_params is None:
            not_opt_params = self.NOT_OPT_PARAMS
        if int_params is None:
            int_params = self.INT_PARAMS
        if param_scales is None:
            param_scales = self.PARAM_SCALES
        if fit_params is None:
            fit_params = self.FIT_PARAMS

        # 入力データからチューニング用パラメータの生成
        tuning_params = self._tuning_param_generation(tuning_params)
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # チューニング対象外パラメータの生成
        not_opt_params = self._not_opt_param_generation(not_opt_params, seed, scoring)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')
        
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        tuning_params = self._add_learner_name(estimator, tuning_params)
        fit_params = self._add_learner_name(estimator, fit_params)
        param_scales = self._add_learner_name(estimator, param_scales)
        not_opt_params = self._add_learner_name(estimator, not_opt_params)
        int_params = self._add_learner_name(estimator, int_params)

        # 引数をプロパティ(インスタンス変数)に反映（チューニング中にパラメータ入力されるため、モデルはdeepcopy）
        self._set_argument_to_property(copy.deepcopy(estimator), tuning_params, cv, seed, scoring, fit_params, not_opt_params, param_scales)
        self.int_params = int_params

        # 引数のスケールを変換(対数スケールパラメータは対数化)
        tuning_params_log = self._log10_conversion(tuning_params, param_scales)

        # ベイズ最適化インスタンス作成
        bo = BayesianOptimization(
            self._bayes_evaluate, tuning_params_log, random_state=seed)
        
        # ここまでに掛かった前処理時間を測定
        self.preprocess_time = time.time() - start
        self.elapsed_times = [self.preprocess_time]
        # ベイズ最適化を実行
        bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq)
        self.elapsed_time = time.time() - start
        self.algo_name = 'bayes-opt'

        # 最適パラメータとスコアを取得
        best_params = bo.max['params']
        self.best_score = bo.max['target']
        # 対数スケールパラメータは10のべき乗をとる
        best_params = self._pow10_conversion(best_params, param_scales)
        # 整数パラメータはint型に変換
        best_params = self._int_conversion(best_params, int_params)
        # パイプライン処理のとき、学習器名を追加
        best_params = self._add_learner_name(estimator, best_params)
        self.best_params = best_params

        # self.estimatorはチューニング時の最終パラメータが入力されているので、estimatorで初期化
        self.estimator = estimator

        # 学習履歴の保持
        params_history_log = bo.space.params  # 対数スケールのままパラメータ履歴が格納されたndarray
        scale_array = np.array([np.full(bo.space.params.shape[0], param_scales[k]) for k in bo.space.keys]).T  # スケール変換用ndarray
        params_hisotry = np.where(scale_array == 'log', np.power(10, params_history_log), params_history_log)  # 対数スケールパラメータは10のべき乗をとる
        self.search_history = pd.DataFrame(params_hisotry, columns=bo.space.keys).to_dict(orient='list')  # パラメータ履歴をDict化
        self.search_history['test_score'] = bo.space.target.tolist()  # スコア履歴を追加
        # 所要時間の保持(elapsed_timesの差分)
        self.search_history['raw_trial_time'] = np.diff(np.array(self.elapsed_times), n=1)

        # 最適モデル保持のため学習（特徴量重要度算出等）
        best_params_refit = {k:v for k,v in best_params.items()}
        best_params_refit.update(not_opt_params)  # 最適化対象以外のパラメータも追加
        if 'random_state' in not_opt_params:
            best_params_refit['random_state'] = self.seed
        best_estimator = copy.deepcopy(estimator)
        best_estimator.set_params(**best_params_refit)
        best_estimator.fit(self.X,
                  self.y,
                  **fit_params
                  )
        self.best_estimator = best_estimator
        
        # MLFlowで記録
        if mlflow_logging == 'log':
            self._mlflow_logging()
        elif mlflow_logging == 'with':
            with mlflow.start_run() as run:
                self._mlflow_logging()
        elif mlflow_logging is not None:
            raise Exception('the "mlflow_logging" argument must be "log", "with" or None')

        # ベイズ最適化で探索した最適パラメータ、評価指標最大値を返す
        return self.best_params, self.best_score


    def _optuna_evaluate(self, trial):
        """
        Optuna最適化時の評価指標算出メソッド
        """
        # パラメータ格納
        params = {}
        for k, v in self.tuning_params.items():
            log = True if self.param_scales[k] == 'log' else False  # 変数のスケールを指定（対数スケールならTrue）
            if k in self.int_params:  # int型のとき
                params[k] = trial.suggest_int(k, v[0], v[1], log=log)
            else:  # float型のとき
                params[k] = trial.suggest_float(k, v[0], v[1], log=log)
        params.update(self.not_opt_params)  # 最適化対象以外のパラメータも追加
        # モデル作成
        estimator = self.estimator
        estimator.set_params(**params)
        # cross_val_scoreでクロスバリデーション
        scores = cross_val_score(estimator, self.X, self.y, cv=self.cv, groups=self.cv_group,
                                 scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        val = scores.mean()
        
        return val

    def optuna_tuning(self, estimator=None, tuning_params=None, cv=None, seed=None, scoring=None,
                      n_trials=None, study_kws=None, optimize_kws=None,
                      not_opt_params=None, int_params=None, param_scales=None, mlflow_logging=None, fit_params=None):
        """
        ベイズ最適化(optuna)

        Parameters
        ----------
        estimator : Dict
            最適化対象の学習器インスタンス。not_opt_paransで指定したパラメータは上書きされるので注意
        tuning_params : Dict[str, Tuple(float, float)]
            最適化対象のパラメータ範囲　{パラメータ名:(パラメータの探索下限,上限),‥}で指定
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはnot_opt_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        n_trials : int
            ベイズ最適化の繰り返し回数
        study_kws : Dict
            optuna.study.create_studyに渡す引数
        optimize_kws : Dict
            optuna.study.Study.optimizeに渡す引数 (n_trials以外)
        not_opt_params : Dict
            最適化対象外のパラメータ一覧
        int_params : List
            整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
        param_scales : Dict
            パラメータ
            のスケール('linear', 'log')(Noneならクラス変数PARAM_SCALESから取得)
        mlflow_logging : str
            MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()
        self.start_time = start

        # 引数非指定時、クラス変数から取得
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        if tuning_params is None:
            tuning_params = self.BAYES_PARAMS
        if cv is None:
            cv = self.CV_NUM
        if seed is None:
            seed = self.SEED
        if scoring is None:
            scoring = self.SCORING
        if n_trials is None:
            n_trials = self.N_ITER_OPTUNA
        if study_kws is None:
            study_kws = {}
        if optimize_kws is None:
            optimize_kws = {}
        if not_opt_params is None:
            not_opt_params = self.NOT_OPT_PARAMS
        if int_params is None:
            int_params = self.INT_PARAMS
        if param_scales is None:
            param_scales = self.PARAM_SCALES
        if fit_params is None:
            fit_params = self.FIT_PARAMS

        # 入力データからチューニング用パラメータの生成
        tuning_params = self._tuning_param_generation(tuning_params)
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # チューニング対象外パラメータの生成
        not_opt_params = self._not_opt_param_generation(not_opt_params, seed, scoring)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')

        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        tuning_params = self._add_learner_name(estimator, tuning_params)
        fit_params = self._add_learner_name(estimator, fit_params)
        param_scales = self._add_learner_name(estimator, param_scales)
        not_opt_params = self._add_learner_name(estimator, not_opt_params)
        int_params = self._add_learner_name(estimator, int_params)

        # 引数をプロパティ(インスタンス変数)に反映（チューニング中にパラメータ入力されるため、モデルはdeepcopy）
        self._set_argument_to_property(copy.deepcopy(estimator), tuning_params, cv, seed, scoring, fit_params, not_opt_params, param_scales)
        self.int_params = int_params

        # ベイズ最適化のインスタンス作成
        if 'sampler' not in study_kws:  # 指定がなければsamplerにTPESamplerを使用
            study_kws['sampler'] = optuna.samplers.TPESampler(seed=seed)
        if 'direction' not in study_kws:  # 指定がなければ最大化方向に最適化
            study_kws['direction'] = 'maximize'
        study = optuna.create_study(**study_kws)

        # ここまでに掛かった前処理時間を測定
        self.preprocess_time = time.time() - start
        # ベイズ最適化を実行
        study.optimize(self._optuna_evaluate, n_trials=n_trials,
                       **optimize_kws)
        self.elapsed_time = time.time() - start
        self.algo_name = 'optuna'

        # 最適パラメータとスコアを取得
        best_params = study.best_trial.params
        self.best_score = study.best_trial.value
        # パイプライン処理のとき、学習器名を追加
        best_params = self._add_learner_name(estimator, best_params)
        self.best_params = best_params

        # self.estimatorはチューニング時の最終パラメータが入力されているので、estimatorで初期化
        self.estimator = estimator

        # 学習履歴の保持
        self.search_history = pd.DataFrame([trial.params for trial in study.trials]).to_dict(orient='list')  # パラメータ履歴をDict化
        self.search_history['test_score'] = [trial.value for trial in study.trials]  # スコア履歴を追加
        # 所要時間の保持
        self.search_history['raw_trial_time'] = [trial.duration.total_seconds() for trial in study.trials]

        # 最適モデル保持のため学習（特徴量重要度算出等）
        best_params_refit = {k:v for k,v in best_params.items()}
        best_params_refit.update(not_opt_params)  # 最適化対象以外のパラメータも追加
        if 'random_state' in not_opt_params:
            best_params_refit['random_state'] = self.seed
        best_estimator = copy.deepcopy(estimator)
        best_estimator.set_params(**best_params_refit)
        best_estimator.fit(self.X,
                  self.y,
                  **fit_params
                  )
        self.best_estimator = best_estimator

        # MLFlowで記録
        if mlflow_logging == 'log':
            self._mlflow_logging()
        elif mlflow_logging == 'with':
            with mlflow.start_run() as run:
                self._mlflow_logging()
        elif mlflow_logging is not None:
            raise Exception('the "mlflow_logging" argument must be "log", "with" or None')
        
        # Optunaで探索した最適パラメータ、チューニング対象外パラメータ、評価指標最大値を返す
        return self.best_params, self.best_score


    def get_feature_importances(self):
        """
        特徴量重要度の取得
        """
        if self.best_estimator is not None:
            return self.best_estimator.feature_importances_
        else:
            return None
    
    def plot_feature_importances(self, ax=None):
        """
        特徴量重要度の表示

        Parameters
        ----------
        ax : 
            表示対象のax（Noneなら新規作成）
        """
        if self.best_estimator is not None:
            # 特徴量重要度の表示
            features = list(reversed(self.x_colnames))
            importances = list(
            reversed(self.best_estimator.feature_importances_.tolist()))
            # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
            if ax is None:
                ax=plt.gca()
            ax.barh(features, importances)
        else:
            raise Exception('please tune parameters before plotting feature importances')
        
    def _plot_validation_curve(self, param_name, scoring, param_values, train_scores, valid_scores,
                               plot_stats='mean', scale='linear', vline=None, rounddigit=3, ax=None):
        """
        検証曲線の描画

        Parameters
        ----------
        param_name : str
            横軸プロット対象のパラメータ名
        scoring : str
            評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        param_values : list
            横軸プロット対象のパラメータの値
        train_scores : ndarray 1d
            縦軸プロット対象の評価指標の値 (学習用データ)
        valid_scores : ndarray 1d
            縦軸プロット対象の評価指標の値 (検証用データ)
        plot_stats : str
            検証曲線としてプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))
        scale : str
            横軸のスケール ('lenear', 'log')
        vline : float
            追加する縦線の位置 (最適パラメータの可視化用、Noneなら非表示)
        addtext : src
            縦線位置以外に追加する文字列 (最適スコアの表示用、Noneなら非表示)
        rounddigit : int
            文字表示の丸め桁数 (vline指定時のみ有効)
        ax : matplotlib.axes._subplots.Axes
            表示対象のax（Noneならplt.plotで1枚ごとにプロット）
        """
        if ax is None:
            ax=plt
        # plot_stats == 'mean'のとき、スコアの平均±標準偏差を表示
        if plot_stats == 'mean':
            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            train_center = train_mean
            train_high = train_mean + train_std
            train_low = train_mean - train_std
            valid_mean = np.mean(valid_scores, axis=1)
            valid_std  = np.std(valid_scores, axis=1)
            valid_center = valid_mean
            valid_high = valid_mean + valid_std
            valid_low = valid_mean - valid_std
        # plot_stats == 'median'のとき、スコアの平均±標準偏差を表示
        elif plot_stats == 'median':
            train_center = np.median(train_scores, axis=1)
            train_high = np.amax(train_scores, axis=1)
            train_low = np.amin(train_scores, axis=1)
            valid_center = np.median(valid_scores, axis=1)
            valid_high = np.amax(valid_scores, axis=1)
            valid_low = np.amin(valid_scores, axis=1)

        # training_scoresをプロット
        ax.plot(param_values, train_center, color='blue', marker='o', markersize=5, label='training score')
        ax.fill_between(param_values, train_high, train_low, alpha=0.15, color='blue')
        # validation_scoresをプロット
        ax.plot(param_values, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
        ax.fill_between(param_values, valid_high, valid_low, alpha=0.15, color='green')

        # 縦線表示
        if vline is not None:
            ax.axvline(x=vline, color='gray')  # 縦線表示
            # 最高スコアの計算
            best_index = np.where(np.array(param_values)==vline)
            best_score = valid_center[best_index][0]
            # 指定桁数で丸める(https://qiita.com/SUZUKI_Masaya/items/7aa26fb242b6cf237fa4)
            vlinetxt = round_digits(vline, rounddigit=rounddigit, method='format')
            scoretxt = round_digits(best_score, rounddigit=rounddigit, method='format')
            ax.text(vline, np.amax(valid_center), f'best_{param_name}={vlinetxt}\nbest_score={scoretxt}',
                    color='black', verticalalignment='bottom', horizontalalignment='left')

        # グラフの表示調整
        ax.grid()
        if isinstance(ax, matplotlib.axes._subplots.Axes):  # axesで1画像プロットするとき
            ax.set_xscale(scale)  # 対数軸 or 通常軸を指定
            ax.set_xlabel(param_name)  # パラメータ名を横軸ラベルに
            ax.set_ylabel(scoring)  # スコア名を縦軸ラベルに
        else:  # pltで別画像プロットするとき
            ax.xscale(scale)  # 対数軸 or 通常軸を指定
            ax.xlabel(param_name)  # パラメータ名を横軸ラベルに
            ax.ylabel(scoring)  # スコア名を縦軸ラベルに
        ax.legend(loc='lower right')  # 凡例

    def _get_validation_curve(self, estimator=None,  validation_curve_params=None, cv=None, seed=None, scoring=None,
                             not_opt_params=None, stable_params=None, fit_params=None):
        """
        検証曲線の取得

        Parameters
        ----------
        estimator : 
            検証曲線対象の学習器インスタンス (Noneならクラス変数から取得)
        validation_curve_params : Dict[str, list]
            検証曲線対象のパラメータ範囲 (Noneならクラス変数から取得)
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード (クロスバリデーション分割用、xgboostの乱数シードはnot_opt_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        not_opt_params : Dict
            検証曲線対象以外のパラメータ一覧 (Noneならクラス変数NOT_OPT_PARAMSから取得)
        stable_params : Dict
            検証曲線対象パラメータの、プロット対象以外のときの値 (Noneならデフォルト値)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 引数非指定時、クラス変数から取得
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        if validation_curve_params is None:
            validation_curve_params = self.VALIDATION_CURVE_PARAMS
        if cv is None:
            cv = self.CV_NUM
        if seed is None:
            seed = self.SEED
        if scoring is None:
            scoring = self.SCORING
        if not_opt_params is None:  # stable_paramsでself.NOT_OPT_PARAMSおよびself.not_opt_paramsが更新されないようDeepCopy
            not_opt_params_valid = copy.deepcopy(self.NOT_OPT_PARAMS)
        else:
            not_opt_params_valid = copy.deepcopy(not_opt_params)
        if fit_params is None:
            fit_params = self.FIT_PARAMS
        
        # 入力データからチューニング用パラメータの生成
        validation_curve_params = self._tuning_param_generation(validation_curve_params)
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # チューニング対象外パラメータの生成
        not_opt_params_valid = self._not_opt_param_generation(not_opt_params_valid, seed, scoring)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        validation_curve_params = self._add_learner_name(estimator, validation_curve_params)
        fit_params = self._add_learner_name(estimator, fit_params)
        not_opt_params_valid = self._add_learner_name(estimator, not_opt_params_valid)

        # stable_paramsが指定されているとき、not_opt_params_validに追加
        if stable_params is not None:
            stable_params = self._add_learner_name(estimator, stable_params)
            not_opt_params_valid.update(stable_params)
        # not_opt_params_validを学習器にセット
        estimator.set_params(**not_opt_params_valid)

        # 検証曲線の取得
        validation_curve_result = {}
        for k, v in validation_curve_params.items():
            train_scores, valid_scores = validation_curve(estimator=estimator,
                                                          X=self.X, y=self.y,
                                                          param_name=k,
                                                          param_range=v,
                                                          fit_params=fit_params,
                                                          groups=self.cv_group,
                                                          cv=cv, scoring=scoring,
                                                          n_jobs=None)
            # 結果をDictに格納
            validation_curve_result[k] = {'param_values': v,
                                        'train_scores': train_scores,
                                        'valid_scores': valid_scores
                                        }
        return validation_curve_result

    def plot_first_validation_curve(self, estimator=None, validation_curve_params=None, cv=None, seed=None, scoring=None,
                                    not_opt_params=None, param_scales=None, plot_stats='mean', axes=None,
                                    fit_params=None):
        """
        初期検討用の検証曲線プロット

        Parameters
        ----------
        estimator : Dict
            検証曲線対象の学習器インスタンス (Noneならクラス変数から取得)
        validation_curve_params : Dict[str, list]
            検証曲線対象のパラメータ範囲 (Noneならクラス変数から取得)
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード (クロスバリデーション分割用、xgboostの乱数シードはnot_opt_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        not_opt_params : Dict
            検証曲線対象以外のパラメータ一覧 (Noneならクラス変数NOT_OPT_PARAMSから取得)
        param_scales : Dict
            検証曲線表示時のスケール('linear', 'log')(Noneならクラス変数PARAM_SCALESから取得)
        plot_stats : Dict
            検証曲線としてプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))
        axes : List[ax]
            使用するaxes (Noneなら1枚ずつ別個にプロット)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 引数非指定時、クラス変数から取得(学習器名追加のため、estimatorも取得)
        if validation_curve_params is None:
            validation_curve_params = self.VALIDATION_CURVE_PARAMS
        if not_opt_params is None:
            not_opt_params = self.NOT_OPT_PARAMS
        if param_scales is None:
            param_scales = self.PARAM_SCALES
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        # 入力データからチューニング用パラメータの生成
        validation_curve_params = self._tuning_param_generation(validation_curve_params)
        # チューニング対象外パラメータの生成
        not_opt_params = self._not_opt_param_generation(not_opt_params, seed, scoring)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        not_opt_params = self._add_learner_name(estimator, not_opt_params)
        validation_curve_params = self._add_learner_name(estimator, validation_curve_params)
        param_scales = self._add_learner_name(estimator, param_scales)

        # 検証曲線を取得
        validation_curve_result = self._get_validation_curve(estimator=estimator,
                            validation_curve_params=validation_curve_params,
                            cv=cv,
                            seed=seed,
                            scoring=scoring,
                            not_opt_params=not_opt_params,
                            stable_params=None,
                            fit_params=fit_params)
        
        # 検証曲線をプロット
        for i, (k, v) in enumerate(validation_curve_result.items()):
            # プロット用のaxを取得(なければNone)
            if axes is None:
                ax = None
            elif len(axes.shape) == 1:  # 1次元axesのとき
                ax = axes[i]
            elif len(axes.shape) == 2:  # 2次元axesのとき
                ax = axes[i // axes.shape[1]][i % axes.shape[1]]
            # 検証曲線表示時のスケールを取得(なければ'linear')
            if k in param_scales.keys():
                scale = param_scales[k]
            else:
                scale = 'linear'
            # 検証曲線をプロット
            self._plot_validation_curve(k, scoring, v['param_values'], v['train_scores'], v['valid_scores'],
                               plot_stats=plot_stats, scale=scale, ax=ax)
            if axes is None:
                plt.show()
        if axes is not None:
            plt.show()

    def plot_best_validation_curve(self, validation_curve_params=None, param_scales=None,
                                   plot_stats='mean', axes=None):
        """
        チューニング後の検証曲線プロット (最適)

        Parameters
        ----------
        validation_curve_params : Dict[str, list]
            検証曲線対象のパラメータ範囲 (Noneならクラス変数から取得)
        param_scales : Dict
            検証曲線表示時のスケール('linear', 'log')(Noneならクラス変数PARAM_SCALESから取得)
        plot_stats : Dict
            検証曲線としてプロットする統計値 ('mean', 'median')
        axes : List[ax]
            使用するaxes (Noneなら1枚ずつ別個にプロット)
        """
        # 引数非指定時、クラス変数から取得
        if validation_curve_params is None:
            validation_curve_params = self.VALIDATION_CURVE_PARAMS
        if param_scales is None:
            param_scales = self.PARAM_SCALES
        # 入力データからチューニング用パラメータの生成
        validation_curve_params = self._tuning_param_generation(validation_curve_params)
        # パイプライン処理のとき、パラメータに学習器名を追加
        validation_curve_params = self._add_learner_name(self.estimator, validation_curve_params)
        param_scales = self._add_learner_name(self.estimator, param_scales)
        
        # 最適化未実施時、エラーを出す
        if self.best_estimator is None:
            raise Exception('please tune parameters before plotting feature importances')
        # self.best_paramsに存在しないキーをvalidaion_curve_paramsから削除
        validation_curve_params = {k: v for k, v in validation_curve_params.items(
                                  ) if k in self.best_params.keys()}
        # validation_curve_paramsにself.best_paramsを追加して昇順ソート
        for k, v in validation_curve_params.items():
            if self.best_params[k] not in v:
                v.append(self.best_params[k])
                v.sort()
        # 検証曲線を取得
        validation_curve_result = self._get_validation_curve(estimator=self.estimator,
                                                            validation_curve_params=validation_curve_params,
                                                            cv=self.cv,
                                                            seed=self.seed,
                                                            scoring=self.scoring, 
                                                            not_opt_params=self.not_opt_params,
                                                            stable_params=self.best_params, 
                                                            fit_params=self.fit_params)
        
        # 検証曲線をプロット
        for i, (k, v) in enumerate(validation_curve_result.items()):
            # プロット用のaxを取得(なければNone)
            if axes is None:
                ax = None
            elif len(axes.shape) == 1:  # 1次元axesのとき
                ax = axes[i]
            elif len(axes.shape) == 2:  # 2次元axesのとき
                ax = axes[i // axes.shape[1]][i % axes.shape[1]]
            # 検証曲線表示時のスケールを取得(なければ'linear')
            if k in param_scales.keys():
                scale = param_scales[k]
            else:
                scale = 'linear'
            # 検証曲線をプロット
            self._plot_validation_curve(k, self.scoring, v['param_values'], v['train_scores'], v['valid_scores'],
                                        plot_stats=plot_stats, scale=scale, vline=self.best_params[k], ax=ax)
            if axes is None:
                plt.show()
        if axes is not None:
            plt.show()

    def _plot_learning_curve(self, estimator=None,  params=None, cv=None, seed=None, scoring=None,
                            plot_stats='mean', rounddigit=3, ax=None, fit_params=None):
        """
        学習曲線の取得

        Parameters
        ----------
        estimator : Dict
            学習曲線対象の学習器インスタンス (Noneならクラス変数から取得)
        params : Dict[str, float]
            学習器に使用するパラメータの値 (Noneならデフォルト)
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード (クロスバリデーション分割用、xgboostの乱数シードはnot_opt_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        plot_stats : str
            検証曲線としてプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))
        rounddigit : int
            文字表示の丸め桁数 (vline指定時のみ有効)
        ax : matplotlib.axes._subplots.Axes
            表示対象のax（Noneならplt.plotで1枚ごとにプロット
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 引数非指定時、クラス変数から取得
        if estimator is None:
            estimator = copy.deepcopy(self.ESTIMATOR)
        if params is None:
            params = {}
        if cv is None:
            cv = self.CV_NUM
        if seed is None:
            seed = self.SEED
        if scoring is None:
            scoring = self.SCORING
        if fit_params is None:
            fit_params = self.FIT_PARAMS
        
        # 乱数シードをparamsに追加
        if 'random_state' in params:
            params['random_state'] = seed
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # GroupKFold、LeaveOneGroupOutのとき、cv_groupが指定されていなければエラーを出す
        if isinstance(cv, GroupKFold) or isinstance(cv, LeaveOneGroupOut):
            if self.cv_group is None:
                raise Exception('"GroupKFold" and "LeaveOneGroupOut" cross validations need "cv_group" argument at the initialization')
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_final_estimator_name(estimator)
        # パイプライン処理のとき、パラメータに学習器名を追加
        params = self._add_learner_name(estimator, params)
        fit_params = self._add_learner_name(estimator, fit_params)
        # paramsを学習器にセット
        estimator.set_params(**params)

        # 学習曲線の取得
        train_sizes, train_scores, valid_scores = learning_curve(estimator=estimator,
                                                                 X=self.X, y=self.y,
                                                                 train_sizes=np.linspace(0.1, 1.0, 10),
                                                                 fit_params=fit_params,
                                                                 groups=self.cv_group,
                                                                 cv=cv, scoring=scoring, n_jobs=None)
        
        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax is None:
            ax=plt.gca()
        # plot_stats == 'mean'のとき、スコアの平均±標準偏差を表示
        if plot_stats == 'mean':
            train_mean = np.mean(train_scores, axis=1)
            train_std  = np.std(train_scores, axis=1)
            train_center = train_mean
            train_high = train_mean + train_std
            train_low = train_mean - train_std
            valid_mean = np.mean(valid_scores, axis=1)
            valid_std  = np.std(valid_scores, axis=1)
            valid_center = valid_mean
            valid_high = valid_mean + valid_std
            valid_low = valid_mean - valid_std
        # plot_stats == 'median'のとき、スコアの平均±標準偏差を表示
        elif plot_stats == 'median':
            train_center = np.median(train_scores, axis=1)
            train_high = np.amax(train_scores, axis=1)
            train_low = np.amin(train_scores, axis=1)
            valid_center = np.median(valid_scores, axis=1)
            valid_high = np.amax(valid_scores, axis=1)
            valid_low = np.amin(valid_scores, axis=1)

        # training_scoresをプロット
        ax.plot(train_sizes, train_center, color='blue', marker='o', markersize=5, label='training score')
        ax.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
        # validation_scoresをプロット
        ax.plot(train_sizes, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
        ax.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')

        # 最高スコアの表示
        best_score = valid_center[len(valid_center) - 1]
        scoretxt = round_digits(best_score, rounddigit=rounddigit, method='format')  # 指定桁数で丸める
        ax.text(np.amax(train_sizes), valid_low[len(valid_low) - 1], f'best_score={scoretxt}',
                color='black', verticalalignment='top', horizontalalignment='right')

        # グラフの表示調整
        ax.grid()
        ax.set_xlabel('Number of training samples')  # パラメータ名を横軸ラベルに
        ax.set_ylabel(scoring)  # スコア名を縦軸ラベルに
        ax.legend(loc='lower right')  # 凡例

    def plot_best_learning_curve(self, plot_stats='mean', ax=None):
        """
        チューニング後の学習曲線プロット (最適パラメータ使用)

        Parameters
        ----------
        plot_stats : Dict
            検証曲線としてプロットする統計値 ('mean', 'median')
        ax : matplotlib.axes._subplots.Axes
            使用するax (Noneならplt.plotで1枚ごとにプロット)
        """
        
        # 最適化未実施時、エラーを出す
        if self.best_estimator is None:
            raise Exception('please tune parameters before plotting feature importances')

        # パラメータに最適パラメータとチューニング対象外パラメータ追加
        params = copy.deepcopy(self.best_params)
        params.update(self.not_opt_params)

        # 学習曲線をプロット
        self._plot_learning_curve(estimator=self.estimator,
                                 params=params,
                                 cv=self.cv,
                                 seed=self.seed,
                                 scoring=self.scoring, 
                                 plot_stats=plot_stats,
                                 ax=ax,
                                 fit_params=self.fit_params)
        plt.show()

    def plot_search_map(self, order=None, pair_n=4, rounddigits_title=3, rank_number=None, rounddigits_score=3,
                            subplot_kws=None, heat_kws=None, scatter_kws=None):
        """
        探索履歴のプロット（グリッドサーチ：ヒートマップ、その他：散布図）

        Parameters
        ----------
        order: List[str]
            表示する順番 (パラメータ名のリストを指定、グラフ横軸 → グラフ縦軸 → 全体縦軸 → 全体横軸の順番)
        pair_n : int, optional
            グリッドサーチ以外の時の図を並べる枚数
        rounddigits_title : int, optional
            グラフタイトルのパラメータ値の丸め桁数 (グリッドサーチ以外)
        rank_number: int, optional
            スコア上位何番目までを文字表示するか
        rounddigits_score : int, optional
            上位スコア表示の丸め桁数
        subplot_kws : Dict, optional
            プロット用のplt.subplots()に渡す引数 (例：figsize)
        heat_kws : Dict, optional
            ヒートマップ用のsns.heatmap()に渡す引数 (グリッドサーチのみ)
        scatter_kws : Dict, optional
            プロット用のplt.subplots()に渡す引数 (グリッドサーチ以外)
        """
        if rank_number is None:
            rank_number = 0
        # subplot_kwsがNoneなら空のdictを入力
        if subplot_kws is None:
            subplot_kws = {}
        # heat_kwsがNoneなら空のdictを入力
        if heat_kws is None:
            heat_kws = {}
        # scatter_kwsがNoneなら空のdictを入力
        if scatter_kws is None:
            scatter_kws = {}
        
        # 最適化未実施時、エラーを出す
        if self.best_estimator is None:
            raise Exception('please tune parameters before plotting feature importances')
        # パラメータと得点の履歴をDataFrame化
        params_cols = list(self.tuning_params.keys()) + ['test_score']
        df_history = pd.DataFrame(self.search_history)[params_cols]

        # パイプライン処理のとき、引数'order'に学習器名を追加
        order = self._add_learner_name(self.estimator, order)

        # ランダムフォレストでパラメータとスコアのfeature_importancesを求める
        rf = RandomForestRegressor(random_state=self.seed)
        params_array = df_history.drop('test_score', axis=1).values
        score_array = df_history['test_score'].values
        rf.fit(params_array, score_array)
        importances = list(rf.feature_importances_)
        importances = pd.Series(importances, name='importances',
                                index=df_history.drop('test_score', axis=1).columns)
        self.param_importances = importances.sort_values(ascending=True)

        ###### パラメータ表示軸の順番を計算 ######
        pair_h, pair_w = 1, 1  # パラメータ数が2以下の時図は1枚のみ
        # パラメータの並び順を指定しているとき、指定したパラメータ以外は使用しない
        if order is not None:
            n_params = len(order)
            new_columns = [param for param in df_history.columns if param in order or param == 'test_score'] # 指定したパラメータ以外を削除した列名リスト
            # 指定したパラメータ名が存在しないとき、エラーを出す
            for param in order:
                if param not in df_history.columns:
                    raise Exception(f'parameter "{param}" is not included in tuning parameters{list(self.tuning_params.keys())}')
            # グリッドサーチのとき、指定したパラメータ以外は最適パラメータのときのスコアを使用（スコア順位もこのフィルタ後のデータから取得するので注意）
            if self.algo_name == 'grid':
                not_order_params = [param for param in df_history.columns if param not in new_columns]
                for param in not_order_params:
                    df_history = df_history[df_history[param] == self.best_params[param]]
                nuniques = df_history.drop('test_score', axis=1).nunique().rename('nuniques')
                if n_params >= 3:  # パラメータ数3以上のときの縦画像枚数
                    pair_h = int(nuniques[order[2]])
                if n_params >= 4:  # パラメータ数4以上のときの横画像枚数
                    pair_w = int(nuniques[order[3]])
            # グリッドサーチ以外のとき、指定したパラメータでグルーピングしたときの最大値を使用
            else:
                df_history = df_history.loc[df_history.groupby(order)['test_score'].idxmax(), :]
                pair_h = pair_n if n_params >= 3 else 1
                pair_w = pair_n if n_params >= 4 else 1

        # パラメータの並び順を指定していないとき、ランダムフォレストのfeature_importancesの並び順とする
        else:
            n_params = len(df_history.columns) - 1
            # グリッドサーチのとき、要素数→feature_importanceの順でソート
            if self.algo_name == 'grid':
                nuniques = df_history.drop('test_score', axis=1).nunique().rename('nuniques')
                df_order = pd.concat([nuniques, importances], axis=1)
                df_order = df_order.sort_values(['nuniques', 'importances'], ascending=[False, False])
                order = df_order.index.tolist()
                if n_params >= 3:  # パラメータ数3以上のときの縦画像枚数
                    pair_h = int(df_order.iloc[2, 0])
                if n_params >= 4:  # パラメータ数4以上のときの横画像枚数
                    pair_w = int(df_order.iloc[3, 0])
            # グリッドサーチ以外の時feature_importancesでソート
            else:
                order = importances.sort_values(ascending=False).index.tolist()
            
        # グリッドサーチ以外でパラメータ数が3以上のとき、図の数と各図の軸範囲を指定
        if self.algo_name != 'grid':
            if n_params >= 3:  # パラメータ数3以上のとき
                pair_h = pair_n
                pair_w = 1
                # 3個目のパラメータを分割する区間
                min_param3 = df_history[order[2]].min()
                max_param3 = df_history[order[2]].max()
                if self.param_scales[order[2]] == 'linear':  # 線形軸のとき
                    separation_param3 = np.linspace(min_param3, max_param3, pair_h + 1)
                else:  # 対数軸のとき
                    separation_param3 = np.logspace(np.log10(min_param3), np.log10(max_param3), pair_h + 1)

            if n_params >= 4:  # パラメータ数4以上のとき
                pair_h = pair_n
                pair_w = pair_n
                # 4個目のパラメータを分割する区間
                min_param4 = df_history[order[3]].min()
                max_param4 = df_history[order[3]].max()
                if self.param_scales[order[3]] == 'linear':  # 線形軸のとき
                    separation_param4 = np.linspace(min_param4, max_param4, pair_w + 1)
                elif self.param_scales[order[3]] == 'log':  # 対数軸のとき
                    separation_param4 = np.logspace(np.log10(min_param4), np.log10(max_param4), pair_w + 1)

        ###### グラフのサイズと軸範囲を指定 ######
        # figsize (全ての図全体のサイズ)指定
        if 'figsize' not in subplot_kws.keys():
            subplot_kws['figsize'] = (pair_w * 6, pair_h * 5)
        # プロット用のaxes作成
        fig, axes = plt.subplots(pair_h, pair_w, **subplot_kws)

        # スコアの上位をdict化して保持
        rank_index  = np.argsort(-df_history['test_score'].values, kind='mergesort')[:rank_number]
        rank_dict = dict(zip(df_history.iloc[rank_index.tolist(), :].index.tolist(), range(rank_number)))

        # グリッドサーチのとき、第5パラメータ以降は最適パラメータを指定して算出
        if self.algo_name == 'grid' and n_params >= 5:
            for i in range(n_params - 4):
                df_history = df_history[df_history[order[i + 4]] == self.best_params[order[i + 4]]]

        # スコアの最大値と最小値を算出（色分けのスケール用）
        score_min = df_history['test_score'].min()
        score_max = df_history['test_score'].max()        
        # 第1＆第2パラメータの設定最大値と最小値を抽出（グラフの軸範囲指定用）
        param1_min = min(self.tuning_params[order[0]])
        param1_max = max(self.tuning_params[order[0]])
        param2_min = min(self.tuning_params[order[1]])
        param2_max = max(self.tuning_params[order[1]])
        # グラフの軸範囲を指定（散布図グラフのみ）
        if self.param_scales[order[0]] == 'linear':
            param1_axis_min = param1_min - 0.1*(param1_max-param1_min)
            param1_axis_max = param1_max + 0.1*(param1_max-param1_min)
        elif self.param_scales[order[0]] == 'log':
            param1_axis_min = param1_min / np.power(10, 0.1*np.log10(param1_max/param1_min))
            param1_axis_max = param1_max * np.power(10, 0.1*np.log10(param1_max/param1_min))
        if self.param_scales[order[1]] == 'linear':
            param2_axis_min = param2_min - 0.1*(param2_max-param2_min)
            param2_axis_max = param2_max + 0.1*(param2_max-param2_min)
        elif self.param_scales[order[1]] == 'log':
            param2_axis_min = param2_min / np.power(10, 0.1*np.log10(param2_max/param2_min))
            param2_axis_max = param2_max * np.power(10, 0.1*np.log10(param2_max/param2_min))

        ###### 図ごとにプロット ######
        # パラメータが1個のとき(1次元折れ線グラフ表示)
        if n_params == 1:
            df_history = df_history.sort_values[order[0]]
            axes.plot(df_history[order[0]], df_history['test_score'])
            axes.set_xscale(self.param_scales[order[0]])  # 対数軸 or 通常軸を指定
            axes.set_xlabel(order[0])  # パラメータ名を横軸ラベルに
            axes.set_ylabel('test_score')  # スコア名を縦軸ラベルに

        # パラメータが2個以上のとき(ヒートマップor散布図表示)
        else:
            for i in range(pair_h):
                for j in range(pair_w):
                    # パラメータが2個のとき (図は1枚のみ)
                    if n_params == 2:
                        ax = axes
                        df_pair = df_history.copy()
                    
                    # パラメータが3個のとき (図はpair_n × 1枚)
                    elif n_params == 3:
                        ax = axes[i]
                        # グリッドサーチのとき、第3パラメータのユニーク値でデータ分割
                        if self.algo_name == 'grid':
                            param3_value = sorted(df_history[order[2]].unique())[i]
                            df_pair = df_history[df_history[order[2]] == param3_value].copy()
                        # グリッドサーチ以外のとき、第3パラメータの値とseparation_param3に基づきデータ分割
                        else:
                            # 第3パラメータ範囲内のみのデータを抽出
                            pair_min3 = separation_param3[i]
                            pair_max3 = separation_param3[i + 1]
                            if i < pair_h - 1:
                                df_pair = df_history[(df_history[order[2]] >= pair_min3) & (
                                                    df_history[order[2]] < pair_max3)].copy()
                            else:
                                df_pair = df_history[df_history[order[2]] >= pair_min3].copy()
                    
                    # パラメータが4個以上のとき (図はpair_n × pair_n枚)
                    elif n_params >= 4:
                        ax = axes[i, j]
                        # グリッドサーチのとき、第3, 第4パラメータのユニーク値でデータ分割
                        if self.algo_name == 'grid':
                            param3_value = sorted(df_history[order[2]].unique())[i]
                            param4_value = sorted(df_history[order[3]].unique())[j]
                            df_pair = df_history[(df_history[order[2]] == param3_value) & (
                                                 df_history[order[3]] == param4_value)].copy()
                        # グリッドサーチ以外のとき、第3, 第4パラメータの値とseparation_param3, separation_param4に基づきデータ分割
                        else:
                            # 第3パラメータ範囲内のみのデータを抽出
                            pair_min3 = separation_param3[i]
                            pair_max3 = separation_param3[i + 1]
                            if i < pair_h - 1:
                                df_pair = df_history[(df_history[order[2]] >= pair_min3) & (
                                                    df_history[order[2]] < pair_max3)].copy()
                            else:
                                df_pair = df_history[df_history[order[2]] >= pair_min3].copy()
                            # 第4パラメータ範囲内のみのデータを抽出
                            pair_min4 = separation_param4[j]
                            pair_max4 = separation_param4[j + 1]
                            if j < pair_w - 1:
                                df_pair = df_pair[(df_pair[order[3]] >= pair_min4) & (
                                                  df_pair[order[3]] < pair_max4)].copy()
                            else:
                                df_pair = df_pair[df_pair[order[3]] >= pair_min4].copy()

                    # グリッドサーチのとき、ヒートマップをプロット
                    if self.algo_name == 'grid':
                        # グリッドデータをピボット化
                        df_pivot = pd.pivot_table(data=df_pair, values='test_score', 
                                                  columns=order[0], index=order[1], aggfunc=np.mean)
                        # 上下軸を反転（元々は上方向が小となっているため）
                        df_pivot = df_pivot.iloc[::-1]
                        # カラーマップとカラーバーのラベルを指定
                        if 'cmap' not in heat_kws.keys():
                            heat_kws['cmap'] = 'YlGn'
                        if 'cbar_kws' not in heat_kws.keys():
                            heat_kws['cbar_kws'] = {'label': 'score'}
                        # ヒートマップをプロット
                        sns.heatmap(df_pivot, ax=ax,
                                    vmin=score_min, vmax=score_max, center=(score_max+score_min)/2,
                                    **heat_kws)
                        # グラフタイトルとして、第3、第4パラメータの名称と範囲を記載
                        if n_params == 3:
                            ax.set_title(f'{order[2]}={param3_value}')
                        if n_params >= 4:
                            ax.set_title(f'{order[2]}={param3_value}\n{order[3]}={param4_value}')

                    # グリッドサーチ以外のとき、散布図をプロット
                    else:
                        # カラーマップと端部色を指定
                        if 'cmap' not in scatter_kws.keys():
                            scatter_kws['cmap'] = 'YlGn'
                        if 'edgecolors' not in scatter_kws.keys():
                            scatter_kws['edgecolors'] = 'lightgrey'
                        sc = ax.scatter(df_pair[order[0]].values, df_pair[order[1]].values,
                                        c=df_pair['test_score'], vmin=score_min, vmax=score_max,
                                        **scatter_kws)
                        cbar = ax.figure.colorbar(sc, None, ax, label='score')  # カラーバー追加
                        ax.set_xscale(self.param_scales[order[0]])  # 第1パラメータの軸スケールを適用
                        ax.set_yscale(self.param_scales[order[1]])  # 第2パラメータの軸スケールを適用
                        ax.set_xlim(param1_axis_min, param1_axis_max)  # X軸表示範囲を第1パラメータ最小値～最大値±αに
                        ax.set_ylim(param2_axis_min, param2_axis_max)  # Y軸表示範囲を第2パラメータ最小値～最大値±αに
                        ax.set_xlabel(order[0])  # X軸ラベル
                        ax.set_ylabel(order[1])  # Y軸ラベル
                        # グラフタイトルとして、第3、第4パラメータの名称と範囲を記載
                        if n_params == 3:
                            ax.set_title(f'{order[2]}={round_digits(pair_min3, rounddigit=rounddigits_title, method="sig")} - {round_digits(pair_max3, rounddigit=rounddigits_title, method="sig")}')
                        if n_params >= 4:
                            ax.set_title(f'{order[2]}={round_digits(pair_min3, rounddigit=rounddigits_title, method="sig")} - {round_digits(pair_max3, rounddigit=rounddigits_title, method="sig")}\n{order[3]}={round_digits(pair_min4, rounddigit=rounddigits_title, method="sig")} - {round_digits(pair_max4, rounddigit=rounddigits_title, method="sig")}')

                    # 誤差上位を文字表示
                    df_rank = df_pair[df_pair.index.isin(rank_dict.keys())]
                    for index, row in df_rank.iterrows():
                        rank_text = f'-<-no{rank_dict[index]+1} score={round_digits(row["test_score"], rounddigit=rounddigits_score, method="sig")}'
                        # グリッドサーチのとき
                        if self.algo_name == 'grid':
                            ax.text(df_pivot.columns.get_loc(row[order[0]]) + 0.5, df_pivot.index.get_loc(row[order[1]]) + 0.5, rank_text, verticalalignment='center', horizontalalignment='left')
                        # グリッドサーチ以外の時
                        else:
                            ax.text(row[order[0]], row[order[1]], rank_text, verticalalignment='center', horizontalalignment='left')

        # 字が重なるのでtight_layoutにする
        plt.tight_layout()
        plt.show()

    def plot_param_importances(self):
        if self.param_importances is None:
            raise Exception('Run "plot_search_map" method before running "plot_param_importances" method')
        plt.barh(self.param_importances.index.values, self.param_importances.values)
        plt.show()

    def get_search_history(self):
        """最適化結果をDataFrameで取得"""
        # 最適化未実施時、エラーを出す
        if self.best_estimator is None:
            raise Exception('please tune parameters before plotting feature importances')
        # パラメータと得点の履歴をDataFrame化
        df_history = pd.DataFrame(self.search_history)
        score_array = df_history['test_score'].values
        time_array = df_history['raw_trial_time'].values
        # その時点までの最大値を取得
        df_history['max_score'] = df_history.index.map(lambda x: max(score_array[:x+1]))
        # その時点までの所要時間を取得(最適化クラスから取得した生値)
        df_history['raw_total_time'] = df_history.index.map(lambda x: sum(time_array[:x+1]))
        # その時点までの所要時間を、elapsed_timeとの比率で補正
        total_tuning_time = np.max(df_history['raw_total_time'])
        df_history['total_time'] = df_history['raw_total_time'] * self.elapsed_time / total_tuning_time

        # DataFrameを返す
        return df_history

    def plot_search_history(self, ax=None, x_axis='index', plot_kws=None):
        """
        探索経過のプロット

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes, optional
            表示対象のax（Noneならplt.plotで1枚ごとにプロット）
        x_axis : str, optional
            横軸の種類('index':試行回数, 'time':経過時間（合計時間での補正値）
        plot_kws : Dict
            プロット用のplt.plotに渡す引数
        """
        # plot_kwsがNoneなら空のdictを入力
        if plot_kws is None:
            plot_kws = {}
        # 描画用axがNoneのとき、matplotlib.pyplot.gca()を使用
        if ax is None:
            ax=plt.gca()
            ax_flg = False
        else:
            ax_flg = True
        
        # パラメータと得点とその最大値の履歴をDataFrame化
        df_history = self.get_search_history()
        if 'color' not in plot_kws:
            plot_kws['color'] = 'red'
        # X軸のデータを取得
        if x_axis == 'index':
            x = df_history.index.tolist()
        elif x_axis == 'time':
            x = [0.0] + df_history['total_time'].tolist()
        # Y軸のデータを取得
        y = df_history['max_score'].tolist()
        if x_axis == 'time':
            y = [np.min(y)] + y

        # グラフをプロット
        ax.plot(x, y, label='training score', **plot_kws)
        ax.set_ylabel('max_test_score')  # Y軸ラベル
        if x_axis == 'index':  # X軸ラベル(試行回数のとき)
            ax.set_xlabel('trials')
        elif x_axis == 'time':
            ax.set_xlabel('time')  # X軸ラベル(経過時間のとき)

        # ax指定していないとき、グラフを表示
        if not ax_flg:
            plt.show()