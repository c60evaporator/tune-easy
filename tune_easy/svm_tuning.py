from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .param_tuning import ParamTuning

class SVMRegressorTuning(ParamTuning):
    """
    Tuning class for SVR

    See ``tune_easy.param_tuning.ParamTuning`` to see API Reference of all methods
    """

    # 共通定数
    _SEED = 42  # デフォルト乱数シード
    _CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (標準化+SVRのパイプライン)
    ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error', etc.)
    _SCORING = 'neg_root_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'kernel': 'rbf'  # カーネルの種類 (基本は'rbf')
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'gamma':[0.001, 0.01, 0.03, 0.1, 0.3, 1, 10],  # RBFカーネルのgamma (小さいと曲率小、大きいと曲率大)
                      'C': [0.01, 0.1, 0.3, 1, 3, 10],  # 正則化項C (小さいと未学習寄り、大きいと過学習寄り)
                      'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.2]  # εチューブの範囲 (大きいほど、誤差関数算出対象=サポートベクターから除外されるデータ点が多くなる)
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 250  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                        'C': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                        'epsilon': [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 100  # BayesianOptimizationの試行数
    INIT_POINTS = 20  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    _ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 300  # Optunaの試行数
    BAYES_PARAMS = {'gamma': (0.001, 10),
                    'C': (0.01, 10),
                    'epsilon': (0, 0.2)
                    }
    INT_PARAMS = []

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
                               'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
                               'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'gamma': 'log',
                    'C': 'log',
                    'epsilon': 'linear'
                    }

class SVMClassifierTuning(ParamTuning):
    """
    Tuning class for SVC

    See ``tune_easy.param_tuning.ParamTuning`` to see API Reference of all methods
    """

    # 共通定数
    _SEED = 42  # デフォルト乱数シード
    _CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (標準化+SVCのパイプライン)
    ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('neg_log_loss', 'roc_auc', 'roc_auc_ovr'など)
    _SCORING = 'neg_log_loss'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'kernel': 'rbf'  # カーネルの種類 (基本は'rbf')
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],  # RBFカーネルのgamma (小さいと曲率小、大きいと曲率大)
                      'C': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]  # 正則化項C (小さいと未学習寄り、大きいと過学習寄り)
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 160  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'gamma': [0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1, 2, 4, 7, 10, 20, 40, 70, 100],
                        'C': [0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1, 2, 4, 7, 10, 20, 40, 70, 100]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 80  # BayesianOptimizationの試行数
    INIT_POINTS = 10  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    _ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 120  # Optunaの試行数
    BAYES_PARAMS = {'gamma': (0.01, 100),
                    'C': (0.01, 100)
                    }
    INT_PARAMS = []

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
                               'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'gamma': 'log',
                    'C': 'log'
                    }
    
    def _not_opt_param_generation(self, src_not_opt_params, seed, scoring):
        """
        チューニング対象外パラメータの生成(seed追加、loglossかつSVRのときのprobablity設定など)

        Parameters
        ----------
        src_not_opt_params : Dict
            処理前のチューニング対象外パラメータ
        seed : int
            乱数シード
        scoring : str
            最適化で最大化する評価指標
        
        """
        # 評価指標がlogloss等クラス確率を必要とするスコアのとき、probabilityとrandom_stateを設定
        if scoring in ['neg_log_loss', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'average_precision']:
            src_not_opt_params['probability'] = True
            src_not_opt_params['random_state'] = seed
        # 乱数シードをnot_opt_paramsのrandom_state引数に追加
        elif 'random_state' in src_not_opt_params:
            src_not_opt_params['random_state'] = seed
        return src_not_opt_params