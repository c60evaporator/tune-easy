from abc import abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization
import time
import numbers
import copy
import pandas as pd
import matplotlib.pyplot as plt

class ParamTuning():
    """
    パラメータチューニング用基底クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス
    CV_MODEL = None
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
     # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = None

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {}

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {}
    CV_PARAMS_GRID.update(NOT_OPT_PARAMS)  # 最適化対象外パラメータを追加

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの繰り返し回数
    CV_PARAMS_RANDOM = {}
    CV_PARAMS_RANDOM.update(NOT_OPT_PARAMS)  # 最適化対象外パラメータを追加

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 100  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 20  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {}
    INT_PARAMS = []  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
    BAYES_NOT_OPT_PARAMS = {k: v[0] for k, v in NOT_OPT_PARAMS.items()}  # ベイズ最適化対象外パラメータ

    # 検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {}

    def __init__(self, X, y, X_colnames, y_colname=None):
        """
        初期化

        Parameters
        ----------
        X : ndarray
            説明変数データ(pandasではなく2次元ndarray)
        y : ndarray
            目的変数データ(ndarray、2次元でも1次元でも可)
        X_colnames : list(str)
            説明変数のフィールド名
        y_colname : str
            目的変数のフィールド名
        """
        if X.shape[1] != len(X_colnames):
            raise Exception('width of X must be equal to length of X_colnames')
        self.X = X
        self.y = y.ravel() # 2次元ndarrayのとき、ravel()で1次元に変換
        self.X_colnames = X_colnames
        self.y_colname = y_colname
        self.tuning_params = None  # チューニング対象のパラメータとその範囲
        self.bayes_not_opt_params = None  # チューニング非対象のパラメータ
        self.int_params = None  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
        self.seed = None  # 乱数シード
        self.cv = None  # クロスバリデーション分割法
        self.cv_model = None  # 最適化対象の学習器インスタンス
        self.learner_name = None  # パイプライン処理時の学習器名称
        self.fit_params = None  # 学習時のパラメータ
        self.best_estimator_ = None  # 最適化された学習モデル
    
    def _train_param_generation(self, src_fit_params):
        """
        入力データから学習時パラメータの生成（例: XGBoostのeval_list）
        通常はデフォルトのままだが、必要であれば継承先でオーバーライド

        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """
        return src_fit_params
    
    def _set_argument_to_property(self, cv_model, tuning_params, cv, seed, scoring, fit_params):
        """
        引数をプロパティ(インスタンス変数)に反映
        """
        self.cv_model = cv_model
        self.tuning_params = tuning_params
        self.cv = cv
        self.seed = seed
        self.scoring = scoring
        self.fit_params = fit_params

    def _add_learner_name(self, model, params):
        """
        パイプライン処理用に、パラメータ名を"学習器名__パラメータ名"に変更
        """
        if isinstance(model, Pipeline):
            # 学習器名が指定されているとき、パラメータ名を変更して処理を進める(既にパラメータ名に'__'が含まれているパラメータは、変更しない)
            if self.learner_name is not None:
                params = {k if '__' in k else f'{self.learner_name}__{k}': v for k, v in params.items()}
            # 指定されていないとき、エラーを返す
            else:
                raise Exception('pipeline model needs "lerner_name" argument')
        return params
    
    def _get_learner_name(self, cv_model):
        steps = cv_model.steps
        self.learner_name = steps[len(steps)-1][0]

    def grid_search_tuning(self, cv_model=None, cv_params=None, cv=None, seed=None, scoring=None, **fit_params):
        """
        グリッドサーチ＋クロスバリデーション

        Parameters
        ----------
        cv_model : Dict
            最適化対象の学習器インスタンス
        cv_params : Dict
            最適化対象のパラメータ一覧
            Pipelineのときは{学習器名__パラメータ名:[パラメータの値候補],‥}で指定する必要あり
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()

        # 引数非指定時、クラス変数から取得
        if cv_model == None:
            cv_model = copy.deepcopy(self.CV_MODEL)
        if cv_params == None:
            cv_params = self.CV_PARAMS_GRID
        if cv == None:
            cv = self.CV_NUM
        if seed == None:
            seed = self.SEED
        if scoring == None:
            scoring = self.SCORING
        if fit_params == {}:
            fit_params = self.FIT_PARAMS

        # 乱数シードをcv_paramsに追加
        if 'random_state' in cv_params:
            cv_params['random_state'] = [seed]
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_learner_name(cv_model)
        # パイプライン処理のとき、パラメータに学習器名を追加
        cv_params = self._add_learner_name(cv_model, cv_params)
        fit_params = self._add_learner_name(cv_model, fit_params)
        
        # 引数をプロパティ(インスタンス変数)に反映
        self._set_argument_to_property(cv_model, cv_params, cv, seed, scoring, fit_params)

        # グリッドサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        gridcv = GridSearchCV(cv_model, cv_params, cv=cv,
                          scoring=scoring, n_jobs=-1)

        # グリッドサーチ実行（学習実行）
        gridcv.fit(self.X,
               self.y,
               **fit_params
               )
        elapsed_time = time.time() - start

        # 最適パラメータの表示
        print('最適パラメータ ' + str(gridcv.best_params_))

        # 最適モデルの保持
        self.best_estimator_ = gridcv.best_estimator_

        # グリッドサーチでの探索結果を返す
        return gridcv.best_params_, gridcv.best_score_, elapsed_time

    def random_search_tuning(self, cv_model=None, cv_params=None, cv=None, seed=None, scoring=None, n_iter=None, **fit_params):
        """
        ランダムサーチ＋クロスバリデーション

        Parameters
        ----------
        cv_model : Dict
            最適化対象の学習器インスタンス
        cv_params : Dict
            最適化対象のパラメータ一覧
            Pipelineのときは{学習器名__パラメータ名:[パラメータの値候補],‥}で指定する必要あり
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        n_iter : int
            ランダムサーチの繰り返し回数
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()

        # 引数非指定時、クラス変数から取得
        if cv_model == None:
            cv_model = copy.deepcopy(self.CV_MODEL)
        if cv_params == None:
            cv_params = self.CV_PARAMS_RANDOM
        if cv == None:
            cv = self.CV_NUM
        if seed == None:
            seed = self.SEED
        if scoring == None:
            scoring = self.SCORING
        if n_iter == None:
            n_iter = self.N_ITER_RANDOM
        if fit_params == {}:
            fit_params = self.FIT_PARAMS
            if 'verbose' in fit_params.keys():
                fit_params['verbose'] = 0

        # 乱数シードをcv_paramsに追加
        if 'random_state' in cv_params:
            cv_params['random_state'] = [seed]
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_learner_name(cv_model)
        # パイプライン処理のとき、パラメータに学習器名を追加
        cv_params = self._add_learner_name(cv_model, cv_params)
        fit_params = self._add_learner_name(cv_model, fit_params)

        # 引数をプロパティ(インスタンス変数)に反映
        self._set_argument_to_property(cv_model, cv_params, cv, seed, scoring, fit_params)

        # ランダムサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        randcv = RandomizedSearchCV(cv_model, cv_params, cv=cv,
                                random_state=seed, n_iter=n_iter, scoring=scoring, n_jobs=-1)

        # ランダムサーチ実行
        randcv.fit(self.X,
               self.y,
               **fit_params
               )
        elapsed_time = time.time() - start

        # 最適パラメータの表示
        print('最適パラメータ ' + str(randcv.best_params_))
        # 最適モデルの保持
        self.best_estimator_ = randcv.best_estimator_

        # ランダムサーチで探索した最適パラメータ、特徴量重要度、所要時間を返す
        return randcv.best_params_, randcv.best_score_, elapsed_time

    @abstractmethod
    def _bayes_evaluate(self):
        """
         ベイズ最適化時の評価指標算出メソッド (継承先でオーバーライドが必須)
        """
        pass

    def _int_conversion(self, bayes_params, int_params):
        """
         ベイズ最適化パラメータのうち、整数のものを四捨五入してint型変換
        """
        bayes_params = {k: round(v) if k in int_params else v for k, v in bayes_params.items()}
        return bayes_params

    def bayes_opt_tuning(self, cv_model=None, bayes_params=None, cv=None, seed=None, scoring=None, n_iter=None, init_points=None, acq=None, bayes_not_opt_params=None, int_params=None, **fit_params):
        """
        ベイズ最適化(bayes_opt)

        Parameters
        ----------
        beyes_params : Dict
            最適化対象のパラメータ範囲　{パラメータ名:(パラメータの探索下限,上限),‥}で指定
            Pipelineのときもkeyに'学習器名__'を追加しないよう注意 (パラメータ名そのものを指定)
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
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
        bayes_not_opt_params : Dict
            最適化対象外のパラメータ一覧
        int_params : List
            整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 処理時間測定
        start = time.time()

        # 引数非指定時、クラス変数から取得
        if cv_model == None:
            cv_model = copy.deepcopy(self.CV_MODEL)
        if bayes_params == None:
            bayes_params = self.BAYES_PARAMS
        if cv == None:
            cv = self.CV_NUM
        if seed == None:
            seed = self.SEED
        if scoring == None:
            scoring = self.SCORING
        if n_iter == None:
            n_iter = self.N_ITER_BAYES
        if init_points == None:
            init_points = self.INIT_POINTS
        if acq == None:
            acq = self.ACQ
        if bayes_not_opt_params == None:
            bayes_not_opt_params = self.BAYES_NOT_OPT_PARAMS
        if int_params == None:
            int_params = self.INT_PARAMS
        if fit_params == {}:
            fit_params = self.FIT_PARAMS

        # 乱数シードをbayes_not_opt_paramsに追加
        if 'random_state' in bayes_not_opt_params:
            bayes_not_opt_params['random_state'] = seed
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_learner_name(cv_model)
        # パイプライン処理のとき、パラメータに学習器名を追加(fit_paramsのみ、調整用パラメータはベイズ最適化用メソッド内で名称変更)
        fit_params = self._add_learner_name(cv_model, fit_params)

        # 引数をプロパティ(インスタンス変数)に反映
        self._set_argument_to_property(cv_model, bayes_params, cv, seed, scoring, fit_params)
        self.bayes_not_opt_params = bayes_not_opt_params
        self.int_params = int_params

        # ベイズ最適化を実行
        bo = BayesianOptimization(
            self._bayes_evaluate, bayes_params, random_state=seed)
        bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq)
        elapsed_time = time.time() - start

        # 評価指標が最大となったときのパラメータを取得
        best_params = bo.max['params']
        # 整数パラメータはint型に変換
        best_params = self._int_conversion(best_params, int_params)
        # 最適化対象以外のパラメータも追加
        best_params.update(self.BAYES_NOT_OPT_PARAMS)
        if 'random_state' in bayes_not_opt_params:
            best_params['random_state'] = self.seed
        # 評価指標の最大値を取得
        best_score = bo.max['target']

        # 最適モデル保持のため学習（特徴量重要度算出等）
        best_model = copy.deepcopy(cv_model)
        best_params = self._add_learner_name(best_model, best_params)
        best_model.set_params(**best_params)
        best_model.fit(self.X,
                  self.y,
                  **fit_params
                  )
        self.best_estimator_ = best_model
        # ベイズ最適化で探索した最適パラメータ、評価指標最大値、所要時間を返す
        return best_params, best_score, elapsed_time


    def get_feature_importances(self, ax=None):
        """
        特徴量重要度の表示と取得

        Parameters
        ----------
        ax : 
            表示対象のax（Noneなら新規作成）
        """
        if self.best_estimator_ is not None:
            # 特徴量重要度の表示
            features = list(reversed(self.X_colnames))
            importances = list(
            reversed(self.best_estimator_.feature_importances_.tolist()))
            if ax == None:
                plt.barh(features, importances)
            else:
                ax.barh(features, importances)
            # 特徴量重要度の
            return self.best_estimator_.feature_importances_
        else:

            return None