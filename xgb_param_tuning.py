import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import numbers
import matplotlib.pyplot as plt
import time
import pandas as pd

# 回帰パラメータチューニング


class XGBRegressorTuning():
    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    SCORING = 'neg_mean_squared_error'  # 最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
    CV_NUM = 5  # 最適化時のクロスバリデーションの分割数
    BOOSTER = 'gbtree'  # 学習時ブースター('gbtree':ツリーモデル, 'dart':ツリーモデル, 'gblinesr':線形モデル)
    OBJECTIVE = 'reg:squarederror'  # 学習時に最小化させる損失関数(デフォルト:'reg:squarederror')
    EVAL_METRIC = 'rmse'  # 学習時のデータ評価指標。基本的にはOBJECTIVEと1対1対応(デフォルト:'rmse')
    EARLY_STOPPING_ROUNDS = 50  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
    # TODO: SCORINGにデフォルトでRMSEが存在しないので、sklearn.metrics.make_scorerで自作の必要あり(https://qiita.com/kimisyo/items/afdf76b9b6fcade640ed)
    # (https://qiita.com/taruto1215/items/2b1f7224a9a4f43906d8)
    # 最適化対象外パラメータ
    NOT_OPT_PARAMS = {'eval_metric': [EVAL_METRIC],  # データの評価指標
                      'objective': [OBJECTIVE],  # 最小化させるべき損失関数
                      'random_state': [SEED],  # 乱数シード
                      'booster': [BOOSTER],  # ブースター
                      }

    # グリッドサーチ用パラメータ(https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f)
    CV_PARAMS_GRID = {'learning_rate': [0.1, 0.3, 0.5],  # 過学習のバランス(高いほど過学習寄り、低いほど汎化寄り)
                      'min_child_weight': [1, 5, 15],  # 葉に割り当てるスコアwiの合計の最小値。これを下回った場合、それ以上の分割を行わない
                      'max_depth': [3, 5, 7],  # 木の深さの最大値
                      'colsample_bytree': [0.5, 0.8, 1.0],  # 列のサブサンプリングを行う比率
                      'subsample': [0.5, 0.8, 1.0]  # 木を構築する前にデータのサブサンプリングを行う比率。1 なら全データ使用、0.5なら半分のデータ使用
                      }
    CV_PARAMS_GRID.update(NOT_OPT_PARAMS)

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 200  # ランダムサーチの繰り返し回数
    CV_PARAMS_RANDOM = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                        'min_child_weight': [1, 3, 5, 7, 9, 11, 13, 15],
                        'max_depth': [3, 4, 5, 6, 7],
                        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        }
    CV_PARAMS_RANDOM.update(NOT_OPT_PARAMS)

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 100  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 20  # 初期観測点の個数(ランダムな探索を何回行うか)
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {'learning_rate': (0.1, 0.5),
                    'min_child_weight': (1, 15),
                    'max_depth': (3, 7),
                    'colsample_bytree': (0.5, 1),
                    'subsample': (0.5, 1)
                    }
    BAYES_NOT_OPT_PARAMS = {k: v[0] for k, v in NOT_OPT_PARAMS.items()}

    def __init__(self, X, y, X_colnames, y_colname=None):
        """
        初期化

        Parameters
        ----------
        X : ndarray
            説明変数データ(pandasではなくndarray)
        y : ndarray
            目的変数データ
        X_colnames : list(str)
            説明変数のフィールド名
        y_colname : str
            目的変数のフィールド名
        """
        if X.shape[1] != len(X_colnames):
            raise Exception('width of X must be equal to length of X_colnames')
        self.X = X
        self.y = y
        self.X_colnames = X_colnames
        self.y_colname = y_colname
        self.tuning_params = None
        self.bayes_not_opt_params = None
        self.seed = None
        self.cv = None
        self.early_stopping_rounds = None

    def grid_search_tuning(self, cv_params=CV_PARAMS_GRID, cv=CV_NUM, seed=SEED, scoring=SCORING, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        """
        グリッドサーチ＋クロスバリデーション

        Parameters
        ----------
        cv_params : dict
            最適化対象のパラメータ一覧
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        """
        # 引数を反映
        cv_params['random_state'] = [seed]
        self.tuning_params = cv_params
        self.seed = seed
        self.cv = cv
        self.scoring = scoring
        self.early_stopping_rounds = early_stopping_rounds
        start = time.time()
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # グリッドサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        gridcv = GridSearchCV(cv_model, cv_params, cv=cv,
                          scoring=scoring, n_jobs=-1)

        # グリッドサーチ実行
        evallist = [(self.X, self.y)]
        gridcv.fit(self.X,
               self.y,
               eval_set=evallist,
               early_stopping_rounds=early_stopping_rounds
               )
        elapsed_time = time.time() - start

        # 最適パラメータの表示
        print('最適パラメータ ' + str(gridcv.best_params_))
        print('変数重要度' + str(gridcv.best_estimator_.feature_importances_))

        # 特徴量重要度の取得と描画
        feature_importances = gridcv.best_estimator_.feature_importances_
        features = list(reversed(self.X_colnames))
        importances = list(
            reversed(gridcv.best_estimator_.feature_importances_.tolist()))
        plt.barh(features, importances)

        # グリッドサーチでの探索結果を返す
        return gridcv.best_params_, gridcv.best_score_, feature_importances, elapsed_time

    def random_search_tuning(self, cv_params=CV_PARAMS_RANDOM, cv=CV_NUM, seed=SEED, scoring=SCORING, early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_iter=N_ITER_RANDOM):
        """
        ランダムサーチ＋クロスバリデーション

        Parameters
        ----------
        cv_params : dict
            最適化対象のパラメータ一覧
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        n_iter : int
            ランダムサーチの繰り返し回数
        """
        # 引数を反映
        cv_params['random_state'] = [seed]
        self.tuning_params = cv_params
        self.seed = seed
        self.cv = cv
        self.scoring = scoring
        self.early_stopping_rounds = early_stopping_rounds
        start = time.time()  # 処理時間測定
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(cv, numbers.Integral):
            cv = KFold(n_splits=cv, shuffle=True, random_state=seed)
        # XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # ランダムサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        randcv = RandomizedSearchCV(cv_model, cv_params, cv=cv,
                                random_state=seed, n_iter=n_iter, scoring=scoring, n_jobs=-1)

        # グリッドサーチ実行
        evallist = [(self.X, self.y)]
        randcv.fit(self.X,
               self.y,
               eval_set=evallist,
               early_stopping_rounds=early_stopping_rounds
               )
        elapsed_time = time.time() - start

        # 最適パラメータの表示
        print('最適パラメータ ' + str(randcv.best_params_))
        print('変数重要度' + str(randcv.best_estimator_.feature_importances_))

        # 特徴量重要度の取得と描画
        feature_importances = randcv.best_estimator_.feature_importances_
        features = list(reversed(self.X_colnames))
        importances = list(
            reversed(randcv.best_estimator_.feature_importances_.tolist()))
        plt.barh(features, importances)

        # ランダムサーチで探索した最適パラメータ、特徴量重要度、所要時間を返す
        return randcv.best_params_, randcv.best_score_, feature_importances, elapsed_time

    # ベイズ最適化時の評価指標算出メソッド(bayes_optは指標を最大化するので、RMSE等のLower is betterな指標は符号を負にして返す)
    def xgb_reg_evaluate(self, learning_rate, min_child_weight, subsample, colsample_bytree, max_depth):
        # 最適化対象のパラメータ
        params = {'learning_rate': learning_rate,
                  'min_child_weight': int(min_child_weight),
                  'max_depth': int(max_depth),
                  'colsample_bytree': colsample_bytree,
                  'subsample': subsample,
                  }
        params.update(self.bayes_not_opt_params)  # 最適化対象以外のパラメータも追加
        params['random_state'] = self.seed
        # XGBoostのモデル作成
        cv_model = xgb.XGBRegressor()
        cv_model.set_params(**params)

        # cross_val_scoreでクロスバリデーション
        fit_params = {'early_stopping_rounds': self.early_stopping_rounds, "eval_set": [
            (self.X, self.y)], 'verbose': 0}
        scores = cross_val_score(cv_model, self.X, self.y, cv=self.cv,
                                 scoring=self.scoring, fit_params=fit_params, n_jobs=-1)
        val = scores.mean()

        # スクラッチでクロスバリデーション
        # scores = []
        # for train, test in self.cv.split(self.X, self.y):
        #     X_train = self.X[train]
        #     y_train = self.y[train]
        #     X_test = self.X[test]
        #     y_test = self.y[test]
        #     cv_model.fit(X_train,
        #              y_train,
        #              eval_set=[(X_train, y_train)],
        #              early_stopping_rounds=self.early_stopping_rounds,
        #              verbose=0
        #              )
        #     pred = cv_model.predict(X_test)
        #     score = r2_score(y_test, pred)
        #     scores.append(score)
        # val = sum(scores)/len(scores)

        return val

    def bayes_opt_tuning(self, beyes_params=BAYES_PARAMS, cv=CV_NUM, seed=SEED, scoring=SCORING, early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_iter=N_ITER_BAYES, init_points=INIT_POINTS, acq=ACQ, bayes_not_opt_params=BAYES_NOT_OPT_PARAMS):
        """
        ベイズ最適化(bayes_opt)

        Parameters
        ----------
        beyes_params : dict
            最適化対象のパラメータ範囲
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
        seed : int
            乱数シード(クロスバリデーション分割用、xgboostの乱数シードはcv_paramsで指定するので注意)
        scoring : str
            最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        n_iter : int
            ベイズ最適化の繰り返し回数
        init_points : int
            初期観測点の個数(ランダムな探索を何回行うか)
        acq : str
            獲得関数('ei', 'pi', 'ucb')
        bayes_not_opt_params : dict
            最適化対象外のパラメータ一覧
        """
        # 引数を反映
        self.tuning_params = beyes_params
        self.bayes_not_opt_params = bayes_not_opt_params
        self.seed = seed
        self.cv = cv
        self.scoring = scoring
        self.early_stopping_rounds = early_stopping_rounds
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if isinstance(self.cv, numbers.Integral):
            self.cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        # ベイズ最適化を実行
        start = time.time()
        xgb_bo = BayesianOptimization(
            self.xgb_reg_evaluate, beyes_params, random_state=seed)
        xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq)
        elapsed_time = time.time() - start
        # 評価指標が最大となったときのパラメータを取得
        best_params = xgb_bo.max['params']
        best_params['min_child_weight'] = int(
            best_params['min_child_weight'])  # 小数で最適化されるのでint型に直す
        best_params['max_depth'] = int(
            best_params['max_depth'])  # 小数で最適化されるのでint型に直す
        # 最適化対象以外のパラメータも追加
        best_params.update(self.BAYES_NOT_OPT_PARAMS)
        best_params['random_state'] = self.seed
        # 評価指標の最大値を取得
        best_score = xgb_bo.max['target']
        # 特徴量重要度算出のため学習
        model = xgb.XGBRegressor()
        model.set_params(**best_params)
        evallist = [(self.X, self.y)]
        model.fit(self.X,
                  self.y,
                  eval_set=evallist,
                  early_stopping_rounds=self.early_stopping_rounds
                  )
        feature_importances = model.feature_importances_
        # ベイズ最適化で探索した最適パラメータ、評価指標最大値、特徴量重要度、所要時間を返す
        return best_params, best_score, feature_importances, elapsed_time

    def multiple_seeds_tuning(self, method, seeds=SEEDS, params=None, cv=CV_NUM, scoring=SCORING, early_stopping_rounds=EARLY_STOPPING_ROUNDS, n_iter=None, init_points=INIT_POINTS, acq=ACQ, bayes_not_opt_params=BAYES_NOT_OPT_PARAMS):
        """
        乱数を変えてループ実行

        Parameters
        ----------
        method : str
            最適化手法('Grid', 'Random', 'Bayes')
        seeds : list(int)
            乱数シードのリスト(クロスバリデーション分割用、xgboostの乱数シードはparams(ベイズ最適化はbayes_not_opt_params)で指定するので注意)
        params : dict
            最適化対象のパラメータ一覧(ベイズ最適化はパラメータ範囲)
        cv : int or KFold
            クロスバリデーション分割法(未指定時 or int入力時はkFoldで分割)
        scoring : str
            最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        n_iter : int
            ランダムサーチ・ベイズ最適化の繰り返し回数(グリッドサーチでは不使用)
        init_points : int
            初期観測点の個数(ベイズ最適化のみ使用)
        acq : str
            獲得関数('ei', 'pi', 'ucb'、ベイズ最適化のみ使用)
        bayes_not_opt_params : dict
            最適化対象外のパラメータ一覧(ベイズ最適化のみ使用)
        """
        # パラメータを指定していない時、デフォルト値を読み込む
        if params == None:
            if method == 'Grid':
                params = self.CV_PARAMS_GRID
            elif method == 'Random':
                params = self.CV_PARAMS_RANDOM
            elif method == 'Bayes':
                params = self.BAYES_PARAMS
        # 探索回数を指定していない時、デフォルト値を読み込む
        if n_iter == None:
            if method == 'Random':
                n_iter = self.N_ITER_RANDOM
            elif method == 'Bayes':
                n_iter = self.N_ITER_BAYES
        
        # 乱数ごとにループして最適化実行
        result_list = []
        for seed in seeds:
            self.__init__(self.X, self.y, self.X_colnames,
                          y_colname=self.y_colname)  # いったん初期化
                    
            # TODO cvに分割法指定しているとき、seedを書き換える必要あり
            if not isinstance(cv, numbers.Integral):
                if cv.random_state is not None:
                    cv.random_state = seed

            # グリッドサーチ
            if method == 'Grid':
                best_params, best_score, feature_importances, elapsed_time = self.grid_search_tuning(
                    cv_params=params, cv=cv, scoring=scoring, seed=seed, early_stopping_rounds=early_stopping_rounds)
            # ランダムサーチ
            elif method == 'Random':
                best_params, best_score, feature_importances, elapsed_time = self.random_search_tuning(
                    cv_params=params, cv=cv, scoring=scoring, seed=seed, early_stopping_rounds=early_stopping_rounds, n_iter=n_iter)
            # ベイズ最適化
            elif method == 'Bayes':
                best_params, best_score, feature_importances, elapsed_time = self.bayes_opt_tuning(
                    beyes_params=params, cv=cv, scoring=scoring, seed=seed, early_stopping_rounds=early_stopping_rounds, n_iter=n_iter, init_points=init_points, acq=acq, bayes_not_opt_params=bayes_not_opt_params)

            # 結果を辞書化
            result_dict = { 'seed' : seed,
                        'method' : method,
                        'elapsed_time' : elapsed_time,
                        'cv_num' : str(cv),
                        'scoring' : scoring,
                        'early_stopping_rounds' : early_stopping_rounds}
            # ランダムサーチ、ベイズ最適化のとき、繰り返し数を辞書に追加
            if method == 'Random' or method == 'Bayes':
                result_dict['n_iter'] = n_iter
            # ベイズ最適化のとき、init_points、acqも辞書に追加
            if method == 'Bayes':
                result_dict['init_points'] = init_points
                result_dict['acq'] = acq
            # 評価指標の最大値
            result_dict['top_score'] = best_score
            # 最適化したパラメータを追加
            result_dict.update({'best_' + k: v for k, v in best_params.items()})
            # 特徴量重要度を追加
            result_dict.update({'inportance_' + k: v for k, v in zip(self.X_colnames, feature_importances)})
            # DataFrameに変換
            df_result = pd.DataFrame([result_dict])
            result_list.append(df_result)
        
        #全ての乱数の結果を合体
        result_all = pd.concat(result_list, ignore_index=True)
        return result_all, params

    # 性能評価(leave_one_out)
