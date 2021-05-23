from param_tuning import ParamTuning
from sklearn.model_selection import cross_val_score
import copy
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RFRegressorTuning(ParamTuning):
    """
    ランダムフォレスト回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (XGBoost)
    CV_MODEL = RandomForestRegressor()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {}
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = 'neg_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'random_state': [SEED],  # 乱数シード
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'n_estimators': [20, 40, 80, 160],  # 木の数
                      'max_features': ['auto', 'sqrt', 'log2'],  # 使用する特徴量の最大数
                      'max_depth': [8, 16, 32, 64],  # 木の深さの最大値
                      'min_samples_split': [2, 4, 8, 16],  # 列のサブサンプリングを行う比率
                      'min_samples_leaf': [1, 4, 8, 16]  # 木を構築する前にデータのサブサンプリングを行う比率。1 なら全データ使用、0.5なら半分のデータ使用
                      }
    CV_PARAMS_GRID.update(NOT_OPT_PARAMS)  # 最適化対象外パラメータを追加

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの繰り返し回数
    CV_PARAMS_RANDOM = {'n_estimators': [20, 40, 60, 80, 160, 240, 320],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'max_depth': [2, 4, 8, 16, 32, 64, 128, 192],
                        'min_samples_split': [2, 4, 6, 8, 12, 16, 20],
                        'min_samples_leaf': [1, 2, 4, 6, 8, 12, 16]
                        }
    CV_PARAMS_RANDOM.update(NOT_OPT_PARAMS)  # 最適化対象外パラメータを追加

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 100  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 20  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {'n_estimators': (20, 320),
                    'max_features': (1, 64),
                    'max_depth': (2, 192),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 20)
                    }
    INT_PARAMS = ['n_estimators', 'max_features', 'max_depth', 'min_samples_split', 'min_samples_leaf']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)
    BAYES_NOT_OPT_PARAMS = {k: v[0] for k, v in NOT_OPT_PARAMS.items()}  # ベイズ最適化対象外パラメータ

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'n_estimators': [20, 40, 60, 80, 160, 240, 320],
                        'max_features': [1, 2, 'auto', 'sqrt', 'log2'],
                        'max_depth': [2, 4, 8, 16, 32, 64, 128, 192],
                        'min_samples_split': [2, 4, 6, 8, 12, 16, 20],
                        'min_samples_leaf': [1, 2, 4, 6, 8, 12, 16]
                        }
    # 検証曲線表示時のスケール('linear', 'log')
    PARAM_SCALES = {'n_estimators': 'log',
                    'max_features': 'linear',
                    'max_depth': 'log',
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
        # src_fit_paramsにmax_featuresが存在するとき、入力データの
        if 'max_features' in src_params:
            x_num = self.X.shape[1]# 特徴量数
            src_max_features = src_params['max_features']
            # リストのとき(ベイズ以外)
            if isinstance(src_max_features, list):
                dst_max_features = []
                for v in src_max_features:
                    if v == 'auto':
                        dst_max_features.append(x_num)
                    elif v == 'sqrt':
                        dst_max_features.append(int(np.sqrt(x_num)))
                    elif v == 'log2':
                        dst_max_features.append(int(np.log2(x_num)))
                    else:
                        dst_max_features.append(v)
                dst_max_features = sorted(list(set(dst_max_features)))
            # リスト以外の時(ベイズ)
            else:
                if src_max_features[0] > x_num:
                    raise Exception('the minimum of "max_features" parameter must be smaller than the number of the features')
                dst_max_features =(src_max_features[0], x_num)
            src_params['max_features'] = dst_max_features
        return src_params

    def _bayes_evaluate(self, **kwargs):
        """
        ベイズ最適化時の評価指標算出メソッド
        """
        # 最適化対象のパラメータ
        params = kwargs
        params = self._int_conversion(params, self.int_params)  # 整数パラメータはint型に変換
        params.update(self.bayes_not_opt_params)  # 最適化対象以外のパラメータも追加
        # ランダムフォレストのモデル作成
        cv_model = copy.deepcopy(self.cv_model)
        cv_model.set_params(**params)

        # cross_val_scoreでクロスバリデーション
        scores = cross_val_score(cv_model, self.X, self.y, cv=self.cv,
                                 scoring=self.scoring, fit_params=self.fit_params, n_jobs=-1)
        val = scores.mean()

        return val