from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from .param_tuning import ParamTuning


class LogisticRegressionTuning(ParamTuning):
    """
    Tuning class for LogisticRegression

    See ``muscle_tuning.param_tuning.ParamTuning`` to see API Reference of all methods
    """

    # 共通定数
    _SEED = 42  # デフォルト乱数シード
    _CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (標準化+ロジスティック回帰のパイプライン)
    ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("logr", LogisticRegression())])
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('neg_log_loss', 'roc_auc', 'roc_auc_ovr'など)
    _SCORING = 'neg_log_loss'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'penalty': 'l2',  # 正則化のペナルティ ('l1', 'l2', 'elasticnet')
                      'solver': 'lbfgs'  # 学習に使用するソルバー ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'C': np.logspace(-2, 3, 21).tolist()  # 正則化項C (小さいと未学習寄り、大きいと過学習寄り)
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 25  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'C': np.logspace(-2, 3, 26).tolist()
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 20  # BayesianOptimizationの試行数
    INIT_POINTS = 5  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    _ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 25  # Optunaの試行数
    BAYES_PARAMS = {'C': (0.01, 1000)
                    }
    INT_PARAMS = []

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'C': np.logspace(-3, 4, 15).tolist()
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'C': 'log',
                    'l1_ratio': 'linear'
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
        # 乱数シードをnot_opt_paramsのrandom_state引数に追加
        if 'random_state' in src_not_opt_params:
            src_not_opt_params['random_state'] = seed
        return src_not_opt_params