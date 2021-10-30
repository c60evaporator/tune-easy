import time
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder

from .param_tuning import ParamTuning
from ._cv_eval_set import cross_val_score_eval_set

class XGBRegressorTuning(ParamTuning):
    """
    XGBoost回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (XGBoost)
    ESTIMATOR = XGBRegressor()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {'verbose': 0,  # 学習中のコマンドライン出力
                  'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
                  'eval_metric': 'rmse'  # early_stopping_roundsの評価指標
                  }
    # 最適化で最大化するデフォルト評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    SCORING = 'neg_mean_squared_error'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'objective': 'reg:squarederror',  # 最小化させるべき損失関数
                      'random_state': SEED,  # 乱数シード
                      'booster': 'gbtree',  # ブースター
                      'n_estimators': 10000  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'learning_rate': [0.05, 0.3],  # 過学習のバランス(高いほど過学習寄り、低いほど汎化寄り）別名eta
                      'min_child_weight': [1, 4, 10],  # 葉に割り当てるスコアwiの合計の最小値。これを下回った場合、それ以上の分割を行わない
                      'max_depth': [2, 9],  # 木の深さの最大値
                      'colsample_bytree': [0.2, 0.5, 1.0],  # 列のサブサンプリングを行う比率
                      'subsample': [0.2, 0.5, 0.8],  # 木を構築する前にデータのサブサンプリングを行う比率。1なら全データ使用、0.5なら半分のデータ使用
                      'reg_lambda': [0.1, 1]
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'learning_rate': [0.05, 0.1, 0.2, 0.3],
                        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
                        'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0.001, 0.003, 0.01, 0.03, 0.1],
                        'reg_lambda': [0.001, 0.003, 0.01, 0.03, 0.1],
                        'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 60  # BayesianOptimizationの試行数
    INIT_POINTS = 10  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 120  # Optunaの試行数
    BAYES_PARAMS = {'learning_rate': (0.05, 0.3),
                    'min_child_weight': (1, 10),
                    'max_depth': (2, 9),
                    'colsample_bytree': (0.2, 1.0),
                    'subsample': (0.2, 1.0),
                    'reg_alpha': (0.001, 0.1),
                    'reg_lambda': (0.001, 0.1),
                    'gamma': (0.0001, 0.1)
                    }
    INT_PARAMS = ['min_child_weight', 'max_depth']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'subsample': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],  # デフォルト
                               'colsample_bytree': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                               'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
                               'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
                               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                               'min_child_weight': [1, 3, 5, 7, 9, 11, 15],
                               'max_depth': [1, 2, 3, 4, 6, 8, 10],
                               'gamma': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'subsample': 'linear',
                    'colsample_bytree': 'linear',
                    'reg_alpha': 'log',
                    'reg_lambda': 'log',
                    'learning_rate': 'log',
                    'min_child_weight': 'linear',
                    'max_depth': 'linear',
                    'gamma': 'log'
                    }

    def _train_param_generation(self, estimator, src_fit_params):
        """
        入力データから学習時パラメータの生成 (eval_set)
        
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
                if self.eval_set_selection is None:  # eval_data_source未指定時、eval_setが入力されていなければeval_data_source='test'とする
                    self.eval_set_selection = 'test'
                if self.eval_set_selection not in ['all', 'train', 'test']:  # eval_data_sourceの指定が間違っていたらエラーを出す
                    raise ValueError('The `eval_set_selection` argument should be "all", "train", or "test" when `eval_set` is not in `fit_params`')
            # src_fit_paramsにeval_setが存在するとき、eval_data_source未指定ならばeval_data_source='original_transformed'とする
            else:
                if self.eval_set_selection is None:
                    self.eval_set_selection = 'original_transformed'

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

        # 全データ or 学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        scores = cross_val_score_eval_set(self.eval_set_selection, estimator, self.X, self.y, cv=self.cv, groups=self.cv_group,
                                          scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
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
        
        # 全データ or 学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        scores = cross_val_score_eval_set(self.eval_set_selection, estimator, self.X, self.y, cv=self.cv, groups=self.cv_group,
                                          scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        val = scores.mean()
        
        return val

class XGBClassifierTuning(ParamTuning):
    """
    XGBoost回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (XGBoost)
    ESTIMATOR = XGBClassifier()
    # 学習時のパラメータのデフォルト値
    FIT_PARAMS = {'verbose': 0,  # 学習中のコマンドライン出力
                  'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
                  'eval_metric': 'logloss'  # early_stopping_roundsの評価指標
                  }
    # 最適化で最大化するデフォルト評価指標('neg_log_loss', 'roc_auc', 'roc_auc_ovr'など)
    SCORING = 'neg_log_loss'

    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'objective': 'binary:logistic',  # 最小化させるべき損失関数
                      'random_state': SEED,  # 乱数シード
                      'booster': 'gbtree',  # ブースター
                      'n_estimators': 10000,  # 最大学習サイクル数（評価指標がearly_stopping_rounds連続で改善しなければ打ち切り）
                      'use_label_encoder': False  # UserWarning防止（The use of label encoder in XGBClassifier is deprecated）
                      }

    # グリッドサーチ用パラメータ
    CV_PARAMS_GRID = {'learning_rate': [0.05, 0.3],  # 過学習のバランス(高いほど過学習寄り、低いほど汎化寄り）別名eta
                      'min_child_weight': [1, 4, 10],  # 葉に割り当てるスコアwiの合計の最小値。これを下回った場合、それ以上の分割を行わない
                      'max_depth': [2, 9],  # 木の深さの最大値
                      'colsample_bytree': [0.2, 0.5, 1.0],  # 列のサブサンプリングを行う比率
                      'subsample': [0.2, 0.5, 0.8],  # 木を構築する前にデータのサブサンプリングを行う比率。1なら全データ使用、0.5なら半分のデータ使用
                      'reg_lambda': [0.1, 1]
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'learning_rate': [0.05, 0.1, 0.2, 0.3],
                        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
                        'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0.001, 0.003, 0.01, 0.03, 0.1],
                        'reg_lambda': [0.001, 0.003, 0.01, 0.03, 0.1],
                        'gamma': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 60  # BayesianOptimizationの試行数
    INIT_POINTS = 10  # BayesianOptimizationの初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # BayesianOptimizationの獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 120  # Optunaの試行数
    BAYES_PARAMS = {'learning_rate': (0.05, 0.3),
                    'min_child_weight': (1, 10),
                    'max_depth': (2, 9),
                    'colsample_bytree': (0.2, 1.0),
                    'subsample': (0.2, 1.0),
                    'reg_alpha': (0.001, 0.1),
                    'reg_lambda': (0.001, 0.1),
                    'gamma': (0.0001, 0.1)
                    }
    INT_PARAMS = ['min_child_weight', 'max_depth']  # 整数型のパラメータのリスト(ベイズ最適化時は都度int型変換する)

    # 範囲選択検証曲線用パラメータ範囲
    VALIDATION_CURVE_PARAMS = {'subsample': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],  # デフォルト
                               'colsample_bytree': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                               'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
                               'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
                               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                               'min_child_weight': [1, 3, 5, 7, 9, 11, 15],
                               'max_depth': [1, 2, 3, 4, 6, 8, 10],
                               'gamma': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
                               }
    # 検証曲線表示等で使用するパラメータのスケール('linear', 'log')
    PARAM_SCALES = {'subsample': 'linear',
                    'colsample_bytree': 'linear',
                    'reg_alpha': 'log',
                    'reg_lambda': 'log',
                    'learning_rate': 'log',
                    'min_child_weight': 'linear',
                    'max_depth': 'linear',
                    'gamma': 'log'
                    }
    
    def _additional_init(self, **kwargs):
        """
        初期化時の追加処理（yのラベルをint化）
        """
        
        # ラベルがstr型ならint化する（str型だとXGBClassifierのuse_label_encoderのWarningが出るため）
        if self.y.dtype.name == 'object':
            print('Your labels (y) are strings (np.object), so encode your labels (y) as integers')
            le = LabelEncoder()
            le.fit(self.y)
            self.y = le.transform(self.y)
        return

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
                if self.eval_set_selection is None:  # eval_data_source未指定時、eval_setが入力されていなければeval_data_source='test'とする
                    self.eval_set_selection = 'test'
                if self.eval_set_selection not in ['all', 'train', 'test']:  # eval_data_sourceの指定が間違っていたらエラーを出す
                    raise ValueError('The `eval_set_selection` argument should be "all", "train", or "test" when `eval_set` is not in `fit_params`')
            # src_fit_paramsにeval_setが存在するとき、eval_data_source未指定ならばeval_data_source='original_transformed'とする
            else:
                if self.eval_set_selection is None:
                    self.eval_set_selection = 'original_transformed'

            # 2クラス分類のときeval_metricはloglossを、多クラス分類のときmloglossを入力
            unique_labels = np.unique(self.y)
            if len(unique_labels) == 2:
                if src_fit_params['eval_metric'] in ['mlogloss']:
                    print('Labels are binary, but "eval_metric" is multiple, so "eval_metric" is set to "logloss"')
                    src_fit_params['eval_metric'] = 'logloss'
            else:
                if src_fit_params['eval_metric'] in ['logloss', 'aucpr']:
                    print('Labels are multiple, but "eval_metric" is binary, so "eval_metric" is set to "mlogloss"')
                    src_fit_params['eval_metric'] = 'mlogloss'

        return src_fit_params

    def _not_opt_param_generation(self, src_not_opt_params, seed, scoring):
        """
        チューニング対象外パラメータの生成(seed追加、クラス数に応じたobjectiveの修正)

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
        
        # 2クラス分類のときobjectiveはloglossを、多クラス分類のときmloglossを入力
        unique_labels = np.unique(self.y)
        if len(unique_labels) == 2:
            if src_not_opt_params['objective'] in ['multi:softmax', 'multi:softprob']:
                print('Labels are binary, but "objective" is multiple, so "objective" is set to "binary:logistic"')
                src_not_opt_params['objective'] = 'binary:logistic'
        else:
            if src_not_opt_params['objective'] in ['binary:logistic']:
                print('Labels are multiple, but "objective" is binary, so "objective" is set to "multi:softmax"')
                src_not_opt_params['objective'] = 'multi:softmax'

        return src_not_opt_params

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

        # 全データ or 学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        scores = cross_val_score_eval_set(self.eval_set_selection, estimator, self.X, self.y, cv=self.cv, groups=self.cv_group,
                                          scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
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
        
        # 全データ or 学習データ or テストデータをeval_setに入力して自作メソッドでクロスバリデーション
        scores = cross_val_score_eval_set(self.eval_set_selection, estimator, self.X, self.y, cv=self.cv, groups=self.cv_group,
                                          scoring=self.scoring, fit_params=self.fit_params, n_jobs=None)
        val = scores.mean()
        
        return val