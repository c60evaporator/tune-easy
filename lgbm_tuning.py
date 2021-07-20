from param_tuning import ParamTuning
from sklearn.model_selection import cross_val_score
from sklearn.metrics import check_scoring
import time
import lightgbm as lgbm

class XGBRegressorTuning(ParamTuning):
    """
    XGBoost回帰チューニング用クラス
    """

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # 最適化時のクロスバリデーションのデフォルト分割数
    
    # 学習器のインスタンス (LightGBM)
    CV_MODEL = lgbm.LGBMRegressor()
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
                      'colsample_bytree': [0.4, 0.7, 1.0],
                      'subsample': [0.4, 1.0],
                      'subsample_freq': [0, 7],
                      'min_child_samples': [2, 10, 50]
                      }

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 600  # ランダムサーチの試行数
    CV_PARAMS_RANDOM = {'reg_alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                        'reg_lambda': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                        'num_leaves': [2, 8, 14, 20, 26, 32, 38, 44, 50],
                        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                        'min_child_samples': [0, 2, 8, 14, 20, 26, 32, 38, 44, 50]
                        }

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 80  # ベイズ最適化の試行数
    INIT_POINTS = 10  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    N_ITER_OPTUNA = 300  # Optunaの試行数
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
    VALIDATION_CURVE_PARAMS = {'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                               'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                               'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
                               'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                               'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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
    
    def _additional_init(self, eval_data_source = 'all', **kwargs):
        """
        初期化時の追加処理
        
        Parameters
        ----------
        eval_data_source : str
            XGBoostのfit_paramsに渡すeval_setのデータ
            'all'なら全データ、'valid'ならテストデータ、'train'なら学習データ
        """
        # eval_dataをテストデータから取得
        self.eval_data_source = eval_data_source
        return

    def _train_param_generation(self, src_fit_params):
        """
        入力データから学習時パラメータの生成 (eval_set)
        
        Parameters
        ----------
        src_fit_params : Dict
            処理前の学習時パラメータ
        """

        # src_fit_paramsにeval_setが存在しないとき、入力データをそのまま追加
        if 'eval_set' not in src_fit_params:
            src_fit_params['eval_set'] =[(self.X, self.y)]

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
        cv_model = self.cv_model
        cv_model.set_params(**params)

        # eval_data_sourceに全データ指定時(cross_val_scoreでクロスバリデーション)
        if self.eval_data_source == 'all':
            scores = cross_val_score(cv_model, self.X, self.y, cv=self.cv,
                                     scoring=self.scoring, fit_params=self.fit_params, n_jobs=-1)
            val = scores.mean()
        # eval_data_sourceに学習orテストデータ指定時(スクラッチでクロスバリデーション)
        else:
            scores = self._scratch_cross_val(cv_model, self.eval_data_source)
            val = sum(scores)/len(scores)
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
        cv_model = self.cv_model
        cv_model.set_params(**params)
        
        # eval_data_sourceに全データ指定時(cross_val_scoreでクロスバリデーション)
        if self.eval_data_source == 'all':
            scores = cross_val_score(cv_model, self.X, self.y, cv=self.cv,
                                    scoring=self.scoring, fit_params=self.fit_params, n_jobs=-1)
            val = scores.mean()

        # eval_data_sourceに学習orテストデータ指定時(スクラッチでクロスバリデーション)
        else:
            scores = self._scratch_cross_val(cv_model, self.eval_data_source)
            val = sum(scores)/len(scores)
        
        return val