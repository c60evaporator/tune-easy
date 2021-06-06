from param_tuning import ParamTuning
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SVMRegressorTuning(ParamTuning):
    """
    サポートベクター回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (標準化+SVRのパイプライン)
    CV_MODEL = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = 'neg_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'kernel': ['rbf'],  # カーネルの種類 (基本は'rbf')
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'gamma':[0.001, 0.01, 0.03, 0.1, 0.3, 1, 10],  # RBFカーネルのgamma (小さいと曲率小、大きいと曲率大)
                      'C': [0.01, 0.1, 0.3, 1, 3, 10],  # 正則化項C (小さいと未学習寄り、大きいと過学習寄り)
                      'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.2]  # εチューブの範囲 (大きいほど、誤差関数算出対象=サポートベクターから除外されるデータ点が多くなる)
                      }
    CV_PARAMS_GRID.update(NOT_OPT_PARAMS)  # 最適化対象外パラメータを追加

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 250  # ランダムサーチの繰り返し回数
    CV_PARAMS_RANDOM = {'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                        'C': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                        'epsilon': [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
                        }
    CV_PARAMS_RANDOM.update(NOT_OPT_PARAMS)  # 最適化対象外パラメータを追加

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 100  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 20  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {'gamma': (0.01, 10),
                    'C': (0.1, 10),
                    'epsilon': (0, 0.2)
                    }
    INT_PARAMS = []
    BAYES_NOT_OPT_PARAMS = {k: v[0] for k, v in NOT_OPT_PARAMS.items()}  # ベイズ最適化対象外パラメータ

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                               'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 1000],
                               'epsilon': [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'gamma': 'log',
                    'C': 'log',
                    'epsilon': 'linear'
                    }

    def _train_param_generation(self, src_fit_params):
        """
        入力データから学習時パラメータの生成 (eval_set)
        
        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """
        return src_fit_params

class SVMClassifierTuning(ParamTuning):
    """
    サポートベクター回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (標準化+SVRのパイプライン)
    CV_MODEL = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('neg_log_loss', 'roc_auc', 'PR-AUC', 'F1-score', 'F1_macro')
    SCORING = 'neg_log_loss'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'kernel': ['rbf'],  # カーネルの種類 (基本は'rbf')
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'gamma': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],  # RBFカーネルのgamma (小さいと曲率小、大きいと曲率大)
                      'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # 正則化項C (小さいと未学習寄り、大きいと過学習寄り)
                      }
    CV_PARAMS_GRID.update(NOT_OPT_PARAMS)

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの繰り返し回数
    CV_PARAMS_RANDOM = {'gamma': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                        'C': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
                        }
    CV_PARAMS_RANDOM.update(NOT_OPT_PARAMS)

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 100  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 20  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {'gamma': (0.01, 10),
                    'C': (0.1, 10)
                    }
    INT_PARAMS = []
    BAYES_NOT_OPT_PARAMS = {k: v[0] for k, v in NOT_OPT_PARAMS.items()}

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'gamma': (0.01, 10),
                               'C': (0.1, 10)
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'gamma': 'log',
                    'C': 'log'
                    }

    def _train_param_generation(self, src_fit_params):
        """
        入力データから学習時パラメータの生成 (eval_set)
        
        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """
        return src_fit_params