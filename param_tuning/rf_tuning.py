import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .param_tuning import ParamTuning

class RFRegressorTuning(ParamTuning):
    """
    ランダムフォレスト回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (RandomForestRegressor)
    ESTIMATOR = RandomForestRegressor()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = 'neg_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'random_state': SEED,  # 乱数シード
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'n_estimators': [20, 80, 160],  # 木の数。デフォルト値100
                      'max_features': ['auto', 'sqrt', 'log2'],  # 使用する特徴量の最大数。デフォルト値auto
                      'max_depth': [2, 8, 32],  # 木の深さの最大値。デフォルト値None（∞）
                      'min_samples_split': [2, 8, 32],  # 各ノードに含まれる最小サンプル数。この条件を満たさなくなるか、max_depthに達するまで分割する。デフォルト値2
                      'min_samples_leaf': [1, 4, 16]  # 各葉に含まれる最小サンプル数。小さいほど回帰線が滑らか（過学習寄り）になる。デフォルト値1
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 150  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                        'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                        'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 10  # BayesianOptimizationの試行数
    INIT_POINTS = 80  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 120  # Optunaの試行数
    BAYES_PARAMS = {'n_estimators': (20, 160),
                    'max_features': (1, 64),  # 最大値は自動でデータの特徴量数に変更されるので注意
                    'max_depth': (2, 32),
                    'min_samples_split': (2, 32),
                    'min_samples_leaf': (1, 16)
                    }
    INT_PARAMS = ['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'n_estimators': [10, 20, 30, 40, 60, 80, 120, 160, 240],
                               'max_features': [1, 2, 'auto', 'sqrt', 'log2'],
                               'max_depth': [1, 2, 4, 6, 8, 12, 16, 24, 32, 48],
                               'min_samples_split': [2, 4, 6, 8, 12, 16, 24, 32, 48],
                               'min_samples_leaf': [1, 2, 4, 6, 8, 12, 16, 24, 32]
                               }
    # 検証曲線表示時のスケール('linear', 'log')
    PARAM_SCALES = {'n_estimators': 'linear',
                    'max_features': 'linear',
                    'max_depth': 'linear',
                    'min_samples_split': 'linear',
                    'min_samples_leaf': 'linear'
                    }

    def _tuning_param_generation(self, src_params):
        """
        入力データから学習時パラメータの生成 (max_features)
        
        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """
        # src_fit_paramsにmax_featuresが存在するとき、入力データの数に合わせてパラメータ候補を追加
        if 'max_features' in src_params:
            x_num = self.X.shape[1]# 特徴量数
            src_max_features = src_params['max_features']
            # リストのとき(ベイズ以外、auto, sqrt, log2を追加)
            if isinstance(src_max_features, list):
                dst_max_features = []
                for v in src_max_features:
                    if v == 'auto':  # autoのとき特徴量数を使用
                        dst_max_features.append(x_num)
                    elif v == 'sqrt':  # sqrtのときsqrt(特徴量数)を使用
                        dst_max_features.append(int(np.sqrt(x_num)))
                    elif v == 'log2':  # log2のときlog2(特徴量数)を使用
                        dst_max_features.append(int(np.log2(x_num)))
                    else:
                        dst_max_features.append(v)
                dst_max_features = sorted(list(set(dst_max_features)))
            # リスト以外の時(ベイズ、最大値が特徴量数を超えていたら特徴量数を設定する)
            else:
                if src_max_features[0] > x_num:
                    raise Exception('The minimum limit of "max_features" parameter must be smaller than the number of the features')
                elif src_max_features[1] > x_num:
                    dst_max_features = (src_max_features[0], x_num)
                else:
                    dst_max_features = src_max_features
            src_params['max_features'] = dst_max_features
        return src_params

class RFClassifierTuning(ParamTuning):
    """
    ランダムフォレスト分類チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (RandomForestClassifier)
    ESTIMATOR = RandomForestClassifier()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('neg_log_loss', 'roc_auc', 'roc_auc_ovr'など)
    SCORING = 'neg_log_loss'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'random_state': SEED,  # 乱数シード
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'n_estimators': [20, 80, 160],  # 木の数。デフォルト値100
                      'max_features': ['auto', 'sqrt', 'log2'],  # 使用する特徴量の最大数。デフォルト値auto
                      'max_depth': [2, 8, 32],  # 木の深さの最大値。デフォルト値None（∞）
                      'min_samples_split': [2, 8, 32],  # 各ノードに含まれる最小サンプル数。この条件を満たさなくなるか、max_depthに達するまで分割する。デフォルト値2
                      'min_samples_leaf': [1, 4, 16]  # 各葉に含まれる最小サンプル数。小さいほど回帰線が滑らか（過学習寄り）になる。デフォルト値1
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 150  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                        'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                        'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 10  # BayesianOptimizationの試行数
    INIT_POINTS = 80  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 120  # Optunaの試行数
    BAYES_PARAMS = {'n_estimators': (20, 160),
                    'max_features': (1, 64),  # 最大値は自動でデータの特徴量数に変更されるので注意
                    'max_depth': (2, 32),
                    'min_samples_split': (2, 32),
                    'min_samples_leaf': (1, 16)
                    }
    INT_PARAMS = ['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'n_estimators': [10, 20, 30, 40, 60, 80, 120, 160, 240],
                               'max_features': [1, 2, 'auto', 'sqrt', 'log2'],
                               'max_depth': [1, 2, 4, 6, 8, 12, 16, 24, 32, 48],
                               'min_samples_split': [2, 4, 6, 8, 12, 16, 24, 32, 48],
                               'min_samples_leaf': [1, 2, 4, 6, 8, 12, 16, 24, 32]
                               }
    # 検証曲線表示時のスケール('linear', 'log')
    PARAM_SCALES = {'n_estimators': 'linear',
                    'max_features': 'linear',
                    'max_depth': 'linear',
                    'min_samples_split': 'linear',
                    'min_samples_leaf': 'linear'
                    }

    def _tuning_param_generation(self, src_params):
        """
        入力データから学習時パラメータの生成 (max_features)
        
        Parameters
        ----------
        src_params : Dict
            処理前の学習時パラメータ
        """
        # src_paramsにmax_featuresが存在するとき、入力データの数に合わせてパラメータ候補を追加
        if 'max_features' in src_params:
            x_num = self.X.shape[1]# 特徴量数
            src_max_features = src_params['max_features']
            # リストのとき(ベイズ以外、auto, sqrt, log2を追加)
            if isinstance(src_max_features, list):
                dst_max_features = []
                for v in src_max_features:
                    if v == 'auto':  # autoのとき特徴量数を使用
                        dst_max_features.append(x_num)
                    elif v == 'sqrt':  # sqrtのときsqrt(特徴量数)を使用
                        dst_max_features.append(int(np.sqrt(x_num)))
                    elif v == 'log2':  # log2のときlog2(特徴量数)を使用
                        dst_max_features.append(int(np.log2(x_num)))
                    else:
                        dst_max_features.append(v)
                dst_max_features = sorted(list(set(dst_max_features)))
            # リスト以外の時(ベイズ、最大値が特徴量数を超えていたら特徴量数を設定する)
            else:
                if src_max_features[0] > x_num:
                    raise Exception('The minimum limit of "max_features" parameter must be smaller than the number of the features')
                elif src_max_features[1] > x_num:
                    dst_max_features = (src_max_features[0], x_num)
                else:
                    dst_max_features = src_max_features
            src_params['max_features'] = dst_max_features
        return src_params