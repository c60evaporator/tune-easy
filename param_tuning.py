from abc import abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, validation_curve, learning_curve
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization
import time
import numbers
import decimal
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
    VALIDATION_CURVE_PARAMS = {}  # パラメータ範囲
    PARAM_SCALES = {}  # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')

    def _additional_init(self, **kwargs):
        """
        初期化時の追加処理 (継承先で記述)
        """
        pass

    def __init__(self, X, y, X_colnames, y_colname=None, **kwargs):
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
        self.best_params = None  # 最適パラメータ
        self.best_score = None  # 最高スコア
        self.elapsed_time = None  # 所要時間
        self.best_estimator_ = None  # 最適化された学習モデル
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
    
    def _get_learner_name(self, model):
        """
        パイプライン処理のとき、最後の要素から学習器名を取得
        """
        if isinstance(model, Pipeline):
            steps = model.steps
            self.learner_name = steps[len(steps)-1][0]
    
    def _round_digits(self, src: float, rounddigit: int = None, method='decimal'):
        """
        指定桁数で小数を丸める

        Parameters
        ----------
        srcdict : Dict[str, float]
            丸め対象のDict
        rounddigit : int
            フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
        method : int
            桁数決定手法（'decimal':小数点以下, 'sig':有効数字(Decimal指定), 'format':formatで有効桁数指定）
        """
        if method == 'decimal':
            return round(src, rounddigit)
        elif method == 'sig':
            with decimal.localcontext() as ctx:
                ctx.prec = rounddigit
                return ctx.create_decimal(src)
        elif method == 'format':
            return '{:.{width}g}'.format(src, width=rounddigit)

    def _scratch_cross_val(self, cv_model, eval_data_source):
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
            cv_model.fit(X_train, y_train,
                            **fit_params)
            scorer = check_scoring(cv_model, self.scoring)
            score = scorer(cv_model, X_test, y_test)

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

    def grid_search_tuning(self, cv_model=None, cv_params=None, cv=None, seed=None, scoring=None, **fit_params):
        """
        グリッドサーチ＋クロスバリデーション

        Parameters
        ----------
        cv_model : Dict
            最適化対象の学習器インスタンス
        cv_params : Dict[str, List[float]]
            最適化対象のパラメータ一覧
            Pipelineのときは{学習器名__パラメータ名:[パラメータの値候補],‥}で指定する必要あり
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
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
        # 入力データからチューニング用パラメータの生成
        cv_params = self._tuning_param_generation(cv_params)
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
        self.elapsed_time = time.time() - start
        
        # 最適パラメータの表示と保持
        print('最適パラメータ ' + str(gridcv.best_params_))
        self.best_params = gridcv.best_params_
        self.best_score = gridcv.best_score_
        # 最適モデルの保持
        self.best_estimator_ = gridcv.best_estimator_

        # グリッドサーチでの探索結果を返す
        return gridcv.best_params_, gridcv.best_score_, self.elapsed_time

    def random_search_tuning(self, cv_model=None, cv_params=None, cv=None, seed=None, scoring=None,
                             n_iter=None, **fit_params):
        """
        ランダムサーチ＋クロスバリデーション

        Parameters
        ----------
        cv_model : Dict
            最適化対象の学習器インスタンス
        cv_params : Dict[str, List[float]]
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
        # 入力データからチューニング用パラメータの生成
        cv_params = self._tuning_param_generation(cv_params)
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
        self.elapsed_time = time.time() - start
        
        # 最適パラメータの表示と保持
        print('最適パラメータ ' + str(randcv.best_params_))
        self.best_params = randcv.best_params_
        self.best_score = randcv.rand_score_
        # 最適モデルの保持
        self.best_estimator_ = randcv.best_estimator_

        # ランダムサーチで探索した最適パラメータ、最適スコア、所要時間を返す
        return randcv.best_params_, randcv.best_score_, self.elapsed_time

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

    def bayes_opt_tuning(self, cv_model=None, bayes_params=None, cv=None, seed=None, scoring=None,
                         n_iter=None, init_points=None, acq=None, bayes_not_opt_params=None, int_params=None, **fit_params):
        """
        ベイズ最適化(bayes_opt)

        Parameters
        ----------
        cv_model : Dict
            最適化対象の学習器インスタンス
        beyes_params : Dict[str, Tuple(float, float)]
            最適化対象のパラメータ範囲　{パラメータ名:(パラメータの探索下限,上限),‥}で指定
            Pipelineのときもkeyに'学習器名__'を追加しないよう注意 (パラメータ名そのものを指定)
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
        # 入力データからチューニング用パラメータの生成
        bayes_params = self._tuning_param_generation(bayes_params)
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
        self.elapsed_time = time.time() - start

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
        self.best_params = best_params
        best_model.set_params(**best_params)
        best_model.fit(self.X,
                  self.y,
                  **fit_params
                  )
        self.best_estimator_ = best_model
        # ベイズ最適化で探索した最適パラメータ、評価指標最大値、所要時間を返す
        return best_params, best_score, self.elapsed_time


    def get_feature_importances(self):
        """
        特徴量重要度の取得
        """
        if self.best_estimator_ is not None:
            return self.best_estimator_.feature_importances_
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
        if self.best_estimator_ is not None:
            # 特徴量重要度の表示
            features = list(reversed(self.X_colnames))
            importances = list(
            reversed(self.best_estimator_.feature_importances_.tolist()))
            if ax == None:
                plt.barh(features, importances)
            else:
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
        if ax == None:
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
            vlinetxt = self._round_digits(vline, rounddigit=rounddigit, method='format')
            scoretxt = self._round_digits(best_score, rounddigit=rounddigit, method='format')
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

    def get_validation_curve(self, cv_model=None,  validation_curve_params=None, cv=None, seed=None, scoring=None,
                             not_opt_params=None, stable_params=None, **fit_params):
        """
        検証曲線の取得

        Parameters
        ----------
        cv_model : 
            検証曲線対象の学習器インスタンス (Noneならクラス変数から取得)
        validation_curve_params : Dict[str, list]
            検証曲線対象のパラメータ範囲 (Noneならクラス変数から取得)
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード (クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        not_opt_params : Dict
            検証曲線対象以外のパラメータ一覧 (Noneならクラス変数BAYES_NOT_OPT_PARAMSから取得)
        stable_params : Dict
            検証曲線対象パラメータの、プロット対象以外のときの値 (Noneならデフォルト値)
        fit_params : Dict
            学習時のパラメータをdict指定(例: XGBoostのearly_stopping_rounds)
            Pipelineのときは{学習器名__パラメータ名:パラメータの値,‥}で指定する必要あり
        """
        # 引数非指定時、クラス変数から取得
        if cv_model == None:
            cv_model = copy.deepcopy(self.CV_MODEL)
        if validation_curve_params == None:
            validation_curve_params = self.VALIDATION_CURVE_PARAMS
        if cv == None:
            cv = self.CV_NUM
        if seed == None:
            seed = self.SEED
        if scoring == None:
            scoring = self.SCORING
        if not_opt_params == None:
            not_opt_params = self.BAYES_NOT_OPT_PARAMS
        if fit_params == {}:
            fit_params = self.FIT_PARAMS
        
        # 乱数シードをnot_opt_paramsに追加
        if 'random_state' in not_opt_params:
            not_opt_params['random_state'] = seed
        # 入力データからチューニング用パラメータの生成
        validation_curve_params = self._tuning_param_generation(validation_curve_params)
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_learner_name(cv_model)
        # パイプライン処理のとき、パラメータに学習器名を追加
        validation_curve_params = self._add_learner_name(cv_model, validation_curve_params)
        fit_params = self._add_learner_name(cv_model, fit_params)
        not_opt_params = self._add_learner_name(cv_model, not_opt_params)

        # stable_paramsが指定されているとき、not_opt_paramsに追加
        if stable_params is not None:
            stable_params = self._add_learner_name(cv_model, stable_params)
            not_opt_params.update(stable_params)
        # not_opt_paramsを学習器にセット
        cv_model.set_params(**not_opt_params)

        # 検証曲線の取得
        validation_curve_result = {}
        for k, v in validation_curve_params.items():
            train_scores, valid_scores = validation_curve(estimator=cv_model,
                                    X=self.X, y=self.y,
                                    param_name=k,
                                    param_range=v,
                                    fit_params=fit_params,
                                    cv=cv, scoring=scoring, n_jobs=-1)
            # 結果をDictに格納
            validation_curve_result[k] = {'param_values': v,
                                        'train_scores': train_scores,
                                        'valid_scores': valid_scores
                                        }
        return validation_curve_result

    def plot_first_validation_curve(self, cv_model=None,  validation_curve_params=None, cv=None, seed=None, scoring=None,
                                    not_opt_params=None, param_scales=None, plot_stats='mean', axes=None,
                                    **fit_params):
        """
        初期検討用の検証曲線プロット

        Parameters
        ----------
        cv_model : Dict
            検証曲線対象の学習器インスタンス (Noneならクラス変数から取得)
        validation_curve_params : Dict[str, list]
            検証曲線対象のパラメータ範囲 (Noneならクラス変数から取得)
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード (クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)
        not_opt_params : Dict
            検証曲線対象以外のパラメータ一覧 (Noneならクラス変数BAYES_NOT_OPT_PARAMSから取得)
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
        # 引数非指定時、クラス変数から取得(学習器名追加のため、cv_modelも取得)
        if validation_curve_params == None:
            validation_curve_params = self.VALIDATION_CURVE_PARAMS
        if param_scales == None:
            param_scales = self.PARAM_SCALES
        if cv_model == None:
            cv_model = copy.deepcopy(self.CV_MODEL)
        # 入力データからチューニング用パラメータの生成
        validation_curve_params = self._tuning_param_generation(validation_curve_params)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_learner_name(cv_model)
        # パイプライン処理のとき、パラメータに学習器名を追加
        validation_curve_params = self._add_learner_name(cv_model, validation_curve_params)
        param_scales = self._add_learner_name(cv_model, param_scales)

        # 検証曲線を取得
        validation_curve_result = self.get_validation_curve(cv_model=cv_model,
                            validation_curve_params=validation_curve_params,
                            cv=cv,
                            seed=seed,
                            scoring=scoring,
                            not_opt_params=not_opt_params,
                            stable_params=None,
                            **fit_params)
        
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
        if validation_curve_params == None:
            validation_curve_params = self.VALIDATION_CURVE_PARAMS
        if param_scales == None:
            param_scales = self.PARAM_SCALES
        # 入力データからチューニング用パラメータの生成
        validation_curve_params = self._tuning_param_generation(validation_curve_params)
        # パイプライン処理のとき、パラメータに学習器名を追加
        validation_curve_params = self._add_learner_name(self.cv_model, validation_curve_params)
        param_scales = self._add_learner_name(self.cv_model, param_scales)
        
        # 最適化未実施時、エラーを出す
        if self.best_estimator_ is None:
            raise Exception('please tune parameters before plotting feature importances')
        # validation_curve_paramsにself.best_paramsを追加して昇順ソート
        for k, v in validation_curve_params.items():
            if self.best_params[k] not in v:
                v.append(self.best_params[k])
                v.sort()
        # self.best_paramsをnot_opt_paramsとstable_paramsに分割
        not_opt_params = {k: v for k, v in self.best_params.items() if k not in validation_curve_params.keys()}
        stable_params = {k: v for k, v in self.best_params.items() if k in validation_curve_params.keys()}
        # 検証曲線を取得
        validation_curve_result = self.get_validation_curve(cv_model=self.cv_model,
                                validation_curve_params=validation_curve_params,
                                cv=self.cv,
                                seed=self.seed,
                                scoring=self.scoring, 
                                not_opt_params=not_opt_params,
                                stable_params=stable_params, 
                                **self.fit_params)
        
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

    def plot_learning_curve(self, cv_model=None,  params=None, cv=None, seed=None, scoring=None,
                            plot_stats='mean', rounddigit=3, ax=None, **fit_params):
        """
        学習曲線の取得

        Parameters
        ----------
        cv_model : Dict
            検証曲線対象の学習器インスタンス (Noneならクラス変数から取得)
        params : Dict[str, float]
            学習器に使用するパラメータの値 (Noneならデフォルト)
        cv : int or KFold
            クロスバリデーション分割法 (Noneのときクラス変数から取得、int入力時はkFoldで分割)
        seed : int
            乱数シード (クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
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
        if cv_model == None:
            cv_model = copy.deepcopy(self.CV_MODEL)
        if params == None:
            params = {}
        if cv == None:
            cv = self.CV_NUM
        if seed == None:
            seed = self.SEED
        if scoring == None:
            scoring = self.SCORING
        if fit_params == {}:
            fit_params = self.FIT_PARAMS
        
        # 乱数シードをparamsに追加
        if 'random_state' in params:
            params['random_state'] = seed
        # 学習データから生成されたパラメータの追加
        fit_params = self._train_param_generation(fit_params)
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # パイプライン処理のとき、最後の要素から学習器名を取得
        self._get_learner_name(cv_model)
        # パイプライン処理のとき、パラメータに学習器名を追加
        params = self._add_learner_name(cv_model, params)
        fit_params = self._add_learner_name(cv_model, fit_params)
        # paramsを学習器にセット
        cv_model.set_params(**params)

        # 学習曲線の取得
        train_sizes, train_scores, valid_scores = learning_curve(estimator=cv_model,
                                                                 X=self.X, y=self.y,
                                                                 train_sizes=np.linspace(0.1, 1.0, 10),
                                                                 fit_params=fit_params,
                                                                 cv=cv, scoring=scoring, n_jobs=-1)
        
        # 描画用axがNoneのとき、matplotlib.pyplotを使用
        if ax == None:
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
        ax.plot(train_sizes, train_center, color='blue', marker='o', markersize=5, label='training score')
        ax.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
        # validation_scoresをプロット
        ax.plot(train_sizes, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
        ax.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')

        # 最高スコアの表示
        best_score = valid_center[len(valid_center) - 1]
        scoretxt = self._round_digits(best_score, rounddigit=rounddigit, method='format')  # 指定桁数で丸める
        ax.text(np.amax(train_sizes), valid_low[len(valid_low) - 1], f'best_score={scoretxt}',
                color='black', verticalalignment='top', horizontalalignment='right')

        # グラフの表示調整
        ax.grid()
        if isinstance(ax, matplotlib.axes._subplots.Axes):  # axesで1画像プロットするとき
            ax.set_xlabel('Number of training samples')  # パラメータ名を横軸ラベルに
            ax.set_ylabel(scoring)  # スコア名を縦軸ラベルに
        else:  # pltで別画像プロットするとき
            ax.xlabel('Number of training samples')  # パラメータ名を横軸ラベルに
            ax.ylabel(scoring)  # スコア名を縦軸ラベルに
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
        if self.best_estimator_ is None:
            raise Exception('please tune parameters before plotting feature importances')

        # 学習曲線をプロット
        self.plot_learning_curve(cv_model=self.cv_model,
                                  params=self.best_params,
                                  cv=self.cv,
                                  seed=self.seed,
                                  scoring=self.scoring, 
                                  plot_stats=plot_stats,
                                  ax=ax,
                                  **self.fit_params)
        plt.show()