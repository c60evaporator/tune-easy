from sklearn.model_selection import cross_val_score
import time
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.pipeline import Pipeline

from .param_tuning import ParamTuning
from .util_methods import cross_val_score_eval_set

class LGBMRegressorTuning(ParamTuning):
    """
    LightGBM回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (LightGBM)
    ESTIMATOR = LGBMRegressor()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {'verbose': 0,  # 学習中のコマンドライン出力
                  'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
                  'eval_metric': 'rmse'  # early_stopping_roundsの評価指標
                  }
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = 'neg_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'objective': 'regression',  # 最小化させるべき損失関数
                      'random_state': SEED,  # 乱数シード
                      'boosting_type': 'gbdt',  # boosting_type
                      'n_estimators': 10000  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'reg_alpha': [0.0001, 0.003, 0.1],
                      'reg_lambda': [0.0001, 0.1],
                      'num_leaves': [2, 10, 50],
                      'colsample_bytree': [0.4, 1.0],
                      'subsample': [0.4, 1.0],
                      'subsample_freq': [0, 7],
                      'min_child_samples': [2, 10, 50]
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 400  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'reg_alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                        'reg_lambda': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                        'num_leaves': [2, 8, 14, 20, 26, 32, 38, 44, 50],
                        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                        'min_child_samples': [0, 2, 8, 14, 20, 26, 32, 38, 44, 50]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 60  # ベイズ最適化の試行数
    INIT_POINTS = 10  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 200  # Optunaの試行数
    BAYES_PARAMS = {'reg_alpha': (0.0001, 0.1),
                    'reg_lambda': (0.0001, 0.1),
                    'num_leaves': (2, 50),
                    'colsample_bytree': (0.4, 1.0),
                    'subsample': (0.4, 1.0),
                    'subsample_freq': (0, 7),
                    'min_child_samples': (0, 50)
                    }
    INT_PARAMS = ['num_leaves', 'subsample_freq', 'min_child_samples']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'reg_alpha': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                               'reg_lambda': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                               'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 192, 256],
                               'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                               'min_child_samples': [0, 2, 5, 10, 20, 30, 50, 70, 100]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'reg_alpha': 'log',
                    'reg_lambda': 'log',
                    'num_leaves': 'linear',
                    'colsample_bytree': 'linear',
                    'subsample': 'linear',
                    'subsample_freq': 'linear',
                    'min_child_samples': 'linear'
                    }

    def _train_param_generation(self, estimator, src_fit_params):
        """
        入力データから学習時パラメータの生成 (eval_set)
        
        Parameters
        ----------
        estimator : Dict
            学習器
        src_fit_params : Dict
            処理前の学習時パラメータ
        """

        # src_fit_paramsにeval_metricが設定されているときのみ以下の処理を実施
        if 'eval_metric' in src_fit_params and src_fit_params['eval_metric'] is not None:
            # src_fit_paramsにeval_setが存在しないとき、入力データをそのまま追加
            if 'eval_set' not in src_fit_params:
                print('There is no "eval_set" in fit_params, so "eval_set" is set to (self.X, self.y)')
                src_fit_params['eval_set'] = [(self.X, self.y)]
            # estimatorがパイプラインかつeval_data_sourceが'original'以外のとき、eval_setに最終学習器以外のtransformを適用
            if isinstance(estimator, Pipeline) and self.eval_data_source != 'original':
                print('The estimator is Pipeline, so X data of "eval_set" is transformed using pipeline')
                X_src = self.X
                transformer = Pipeline([step for i, step in enumerate(estimator.steps) if i < len(estimator) - 1])
                X_dst = transformer.fit_transform(X_src)
            else:
                X_dst = self.X
            src_fit_params['eval_set'] = [(X_dst, self.y)]

        return src_fit_params

    def _bayes_evaluate(self, **kwargs):
        """
         ベイズ最適化時の評価指標算出メソッド
        """
        # 最適化対象のパラメータ
        params = kwargs
        params = self._pow10_conversion(params, self.param_scales)  # 対数パラメータは10のべき乗に変換
        params = self._int_conversion(params, self.int_params)  # 整数パラメータはint型に変換
        params.update(self.not_opt_params)  # 最適化対象以外のパラメータも追加
        # XGBoostのモデル作成
        estimator = self.estimator
        estimator.set_params(**params)

        # eval_data_source = 'all', 'original', or 'original_transferred'のとき、cross_val_scoreメソッドでクロスバリデーション
        if self.eval_data_source in ['all', 'original', 'original_transferred']:
            scores = cross_val_score(estimator, self.X, self.y, cv=self.cv,
                                     scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        # eval_data_source = 'train' or 'test'のとき、学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        elif self.eval_data_source in ['train', 'test']:
            scores = cross_val_score_eval_set(self.eval_data_source, estimator, self.X, self.y, cv=self.cv,
                                              scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        else:
            raise Exception('the "eval_data_source" argument must be "all", "test", "train", "original", or "original_transferred"')
        val = scores.mean()
        # 所要時間測定
        self.elapsed_times.append(time.time() - self.start_time)

        return val

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
        # XGBoostのモデル作成
        estimator = self.estimator
        estimator.set_params(**params)
        
        # eval_data_source = 'all', 'original', or 'original_transferred'のとき、cross_val_scoreメソッドでクロスバリデーション
        if self.eval_data_source in ['all', 'original', 'original_transferred']:
            scores = cross_val_score(estimator, self.X, self.y, cv=self.cv,
                                     scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        # eval_data_source = 'train' or 'test'のとき、学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        elif self.eval_data_source in ['train', 'test']:
            scores = cross_val_score_eval_set(self.eval_data_source, estimator, self.X, self.y, cv=self.cv,
                                              scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        else:
            raise Exception('the "eval_data_source" argument must be "all", "test", "train", "original", or "original_transferred"')
        val = scores.mean()
        
        return val

class LGBMClassifierTuning(ParamTuning):
    """
    LightGBM分類チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (LightGBM)
    ESTIMATOR = LGBMClassifier()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {'verbose': 0,  # 学習中のコマンドライン出力
                  'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
                  'eval_metric': 'binary_logloss'  # early_stopping_roundsの評価指標
                  }
    # 最適化で最大化するデフォルト評価指標('neg_log_loss', 'roc_auc', 'roc_auc_ovr'など)
    SCORING = 'neg_log_loss'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'objective': 'binary',  # 最小化させるべき損失関数
                      'random_state': SEED,  # 乱数シード
                      'boosting_type': 'gbdt',  # boosting_type
                      'n_estimators': 10000  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'reg_alpha': [0.0001, 0.003, 0.1],
                      'reg_lambda': [0.0001, 0.1],
                      'num_leaves': [2, 10, 50],
                      'colsample_bytree': [0.4, 1.0],
                      'subsample': [0.4, 1.0],
                      'subsample_freq': [0, 7],
                      'min_child_samples': [2, 10, 50]
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 400  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'reg_alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                        'reg_lambda': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                        'num_leaves': [2, 8, 14, 20, 26, 32, 38, 44, 50],
                        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                        'min_child_samples': [0, 2, 8, 14, 20, 26, 32, 38, 44, 50]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 60  # ベイズ最適化の試行数
    INIT_POINTS = 10  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 200  # Optunaの試行数
    BAYES_PARAMS = {'reg_alpha': (0.0001, 0.1),
                    'reg_lambda': (0.0001, 0.1),
                    'num_leaves': (2, 50),
                    'colsample_bytree': (0.4, 1.0),
                    'subsample': (0.4, 1.0),
                    'subsample_freq': (0, 7),
                    'min_child_samples': (0, 50)
                    }
    INT_PARAMS = ['num_leaves', 'subsample_freq', 'min_child_samples']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'reg_alpha': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                               'reg_lambda': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                               'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 192, 256],
                               'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                               'min_child_samples': [0, 2, 5, 10, 20, 30, 50, 70, 100]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'reg_alpha': 'log',
                    'reg_lambda': 'log',
                    'num_leaves': 'linear',
                    'colsample_bytree': 'linear',
                    'subsample': 'linear',
                    'subsample_freq': 'linear',
                    'min_child_samples': 'linear'
                    }

    def _train_param_generation(self, estimator, src_fit_params):
        """
        入力データから学習時パラメータの生成 (eval_setおよびクラス数に応じたeval_metricの修正)
        
        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """

        # src_fit_paramsにeval_metricが設定されているときのみ以下の処理を実施
        if 'eval_metric' in src_fit_params and src_fit_params['eval_metric'] is not None:
            # src_fit_paramsにeval_setが存在しないとき、入力データをそのまま追加
            if 'eval_set' not in src_fit_params:
                print('There is no "eval_set" in fit_params, so "eval_set" is set to (self.X, self.y)')
                src_fit_params['eval_set'] = [(self.X, self.y)]
            # estimatorがパイプラインかつeval_data_sourceが'original'以外のとき、eval_setに最終学習器以外のtransformを適用
            if isinstance(estimator, Pipeline) and self.eval_data_source != 'original':
                print('The estimator is Pipeline, so X data of "eval_set" is transformed using pipeline')
                X_src = self.X
                transformer = Pipeline([step for i, step in enumerate(estimator.steps) if i < len(estimator) - 1])
                X_dst = transformer.fit_transform(X_src)
            else:
                X_dst = self.X
            src_fit_params['eval_set'] = [(X_dst, self.y)]

            # 2クラス分類のときeval_metricはbinary_logloss or binary_errorを、多クラス分類のときmulti_logloss or multi_errorを入力
            unique_labels = np.unique(self.y)
            if len(unique_labels) == 2:
                if src_fit_params['eval_metric'] in ['multi_logloss', 'multi_error']:
                    print('Labels are binary, but "eval_metric" is multiple, so "eval_metric" is set to "binary_logloss"')
                    src_fit_params['eval_metric'] = 'binary_logloss'
            else:
                if src_fit_params['eval_metric'] in ['binary_logloss', 'binary_error']:
                    print('Labels are multiple, but "eval_metric" is binary, so "eval_metric" is set to "multi_logloss"')
                    src_fit_params['eval_metric'] = 'multi_logloss'

        return src_fit_params

    def _bayes_evaluate(self, **kwargs):
        """
         ベイズ最適化時の評価指標算出メソッド
        """
        # 最適化対象のパラメータ
        params = kwargs
        params = self._pow10_conversion(params, self.param_scales)  # 対数パラメータは10のべき乗に変換
        params = self._int_conversion(params, self.int_params)  # 整数パラメータはint型に変換
        params.update(self.not_opt_params)  # 最適化対象以外のパラメータも追加
        # XGBoostのモデル作成
        estimator = self.estimator
        estimator.set_params(**params)

        # eval_data_source = 'all', 'original', or 'original_transferred'のとき、cross_val_scoreメソッドでクロスバリデーション
        if self.eval_data_source in ['all', 'original', 'original_transferred']:
            scores = cross_val_score(estimator, self.X, self.y, cv=self.cv,
                                     scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        # eval_data_source = 'train' or 'test'のとき、学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        elif self.eval_data_source in ['train', 'test']:
            scores = cross_val_score_eval_set(self.eval_data_source, estimator, self.X, self.y, cv=self.cv,
                                              scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        else:
            raise Exception('the "eval_data_source" argument must be "all", "test", "train", "original", or "original_transferred"')
        val = scores.mean()
        # 所要時間測定
        self.elapsed_times.append(time.time() - self.start_time)

        return val

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
        # XGBoostのモデル作成
        estimator = self.estimator
        estimator.set_params(**params)
        
        # eval_data_source = 'all', 'original', or 'original_transferred'のとき、cross_val_scoreメソッドでクロスバリデーション
        if self.eval_data_source in ['all', 'original', 'original_transferred']:
            scores = cross_val_score(estimator, self.X, self.y, cv=self.cv,
                                     scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        # eval_data_source = 'train' or 'test'のとき、学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        elif self.eval_data_source in ['train', 'test']:
            scores = cross_val_score_eval_set(self.eval_data_source, estimator, self.X, self.y, cv=self.cv,
                                              scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        else:
            raise Exception('the "eval_data_source" argument must be "all", "test", "train", "original", or "original_transferred"')
        val = scores.mean()
        
        return val

    def _not_opt_param_generation(self, src_not_opt_params, seed, scoring):
        """
        チューニング対象外パラメータの生成(seed追加＆)
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
        # 2クラス分類のときobjectiveはbinaryを、多クラス分類のときmulticlassを入力
        unique_labels = np.unique(self.y)
        if len(unique_labels) == 2:
            if src_not_opt_params['objective'] in ['multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr']:
                print('Labels are binary, but "objective" is multiple, so "objective" is set to "binary"')
                src_not_opt_params['objective'] = 'binary'
        else:
            if src_not_opt_params['objective'] in ['binary']:
                print('Labels are multiple, but "objective" is binary, so "objective" is set to "multiclass"')
                src_not_opt_params['objective'] = 'multiclass'

        # 乱数シードをnot_opt_paramsのrandom_state引数に追加
        if 'random_state' in src_not_opt_params:
            src_not_opt_params['random_state'] = seed
        return src_not_opt_params