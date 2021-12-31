from sklearn.linear_model import LinearRegression

from .param_tuning import ParamTuning

class LinearRegressionTuning(ParamTuning):
    """
    Dummy class for LinearRegression

    See ``tune_easy.param_tuning.ParamTuning`` to see API Reference of all methods
    """

    # 共通定数
    _SEED = 42  # デフォルト乱数シード
    _CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (標準化+SVRのパイプライン)
    ESTIMATOR = LinearRegression()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', etc.)
    _SCORING = 'neg_root_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {}

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'fit_intercept': [True]}  # 仮でfit_interceptを入れる

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 1  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'fit_intercept': [True]}

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 1  # BayesianOptimizationの試行数
    INIT_POINTS = 1  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    _ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 1  # Optunaの試行数
    BAYES_PARAMS = {}
    INT_PARAMS = []

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {}
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {}

    # bayes_opt_tuningでもランダムサーチを実行
    def bayes_opt_tuning(self, estimator=None, tuning_params=None, cv=None, seed=None, scoring=None,
                         n_iter=None, init_points=None, acq=None,
                         not_opt_params=None, int_params=None, param_scales=None, mlflow_logging=None, fit_params=None):
        self.random_search_tuning(estimator=estimator, tuning_params=tuning_params, cv=cv, seed=seed, scoring=scoring,
                                  n_iter=n_iter,
                                  not_opt_params=not_opt_params, param_scales=param_scales, mlflow_logging=mlflow_logging, rand_kws=None, fit_params=fit_params)
    
    # optuna_tuningではランダムサーチを実行
    def optuna_tuning(self, estimator=None, tuning_params=None, cv=None, seed=None, scoring=None,
                      n_trials=None, study_kws=None, optimize_kws=None,
                      not_opt_params=None, int_params=None, param_scales=None, mlflow_logging=None, fit_params=None):
        self.random_search_tuning(estimator=estimator, tuning_params=tuning_params, cv=cv, seed=seed, scoring=scoring,
                                  n_iter=n_trials,
                                  not_opt_params=not_opt_params, param_scales=param_scales, mlflow_logging=mlflow_logging, rand_kws=None, fit_params=fit_params)