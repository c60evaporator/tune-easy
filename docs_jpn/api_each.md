# 詳細チューニング API仕様
詳細チューニング実施用クラスのAPI仕様を記載します ([English version](https://c60evaporator.github.io/muscle-tuning/each_estimators.html))

チューニングの実行手順は[こちら](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_each.md)を参照ください

[サンプルコードはこちらにアップロードしています](https://github.com/c60evaporator/muscle-tuning/tree/master/examples/method_examples)

# クラス一覧
使用したい機械学習アルゴリズム（学習器）に合わせて適切なクラスを選択してください

※ここにない学習器の追加希望があれば、[Issues](https://github.com/c60evaporator/muscle-tuning/issues)または[私のTwitter](https://twitter.com/c60evaporator)まで連絡下さい

- **分類**

|クラス名|パッケージ名|概要|デフォルトパラメータのリンク|
|---|---|---|---|
|LGBMClassifierTuning|lgbm_tuning.py|LightGBM分類のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.lgbm_tuning.LGBMClassifierTuning)|
|XGBClassifierTuning|xgb_tuning.py|XGBoost分類のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.xgb_tuning.XGBClassifierTuning)|
|SVMClassifierTuning|svm_tuning.py|サポートベクターマシン分類のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.svm_tuning.SVMClassifierTuning)|
|RFClassifierTuning|rf_tuning.py|ランダムフォレスト分類のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.rf_tuning.RFClassifierTuning)|
|LogisticRegressionTuning|logisticregression_tuning.py|ロジスティック回帰分類のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.logisticregression_tuning.LogisticRegressionTuning)|

- **回帰**

|クラス名|パッケージ名|概要|デフォルトパラメータのリンク|
|---|---|---|---|
|LGBMRegressorTuning|lgbm_tuning.py|LightGBM回帰のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.lgbm_tuning.LGBMRegressorTuning)|
|XGBRegressorTuning|xgb_tuning.py|XGBoost回帰のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.xgb_tuning.XGBRegressorTuning)|
|SVMRegressorTuning|svm_tuning.py|サポートベクター回帰のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.svm_tuning.SVMRegressorTuning)|
|RFRegressorTuning|rf_tuning.py|ランダムフォレスト回帰のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.rf_tuning.RFRegressorTuning)|
|ElasticNetTuning|elasticnet_tuning.py|ElasticNet回帰のパラメータチューニング用クラス|[リンク](https://c60evaporator.github.io/muscle-tuning/each_estimators.html#muscle_tuning.elasticnet_tuning.ElasticNetTuning)|

<br>

## クラス初期化
上記クラスは、以下のように初期化(`__init__()`メソッド)します

### 引数
初期化の引数は以下のようになります
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|X|必須|np.ndarray|-|説明変数データ (2次元のndarray)|
|y|必須|np.ndarray|-|目的変数データ (1次元or2次元のndarray)|
|x_colnames|必須|list[str]|-|説明変数のフィールド名のリスト|
|y_colname|オプション|str|None|目的変数のフィールド名|
|cv_group|オプション|str|None|GroupKFold、LeaveOneGroupOutのグルーピング対象データ|
|eval_data_source|オプション|{'all', 'test', 'train', 'original', 'original_transformed'}|'all'|XGBoost, LightGBMにおける`fit_params['eval_set']`の指定方法※|

#### ※ eval_data_sourceの指定値による、eval_setに入るデータの変化
  - 'all' : eval_set = [(self.X, self.y)]
  - 'test' : eval_set = [(self.X, self.y)]のうちクロスバリデーションのテストデータ
  - 'train' : eval_set = [(self.X, self.y)]のうちクロスバリデーションの学習データ
  - 'original' : チューニング用メソッドの`fit_params`引数に明示的に与えた'eval_set'
  - 'original_transformed' : チューニング用メソッドの`fit_params`引数に明示的に与えた'eval_set' (estimatorがパイプラインの時、前処理を自動適用)

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/init.py)
#### オプション引数指定なしで初期化
LightGBM回帰におけるクラス初期化実行例

```python
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
###### クラス初期化 ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
```

#### LeaveOneGroupOutでクロスバリデーションしたいとき
SVRにおける引数指定例

```python
from param_tuning import XGBRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
###### クラス初期化 ######
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY,  # 必須引数
                            cv_group=df_reg['ward_after'].values)  # グルーピング対象データ (大阪都構想の区)
```

#### 検証データをfit_paramsのeval_setに使用したいとき
デフォルトではeval_set (early_stopping_roundの判定に使用するデータ)は全てのデータ (self.X, self.y)を使用しますが、eval_data_source='valid'を指定するとクロスバリデーションの検証用データのみを使用します

```python
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
X = df_reg[USE_EXPLANATORY].values
y = df_reg[TARGET_VARIABLE].values
###### クラス初期化 ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY,
                             eval_data_source='valid')  # eval_setの指定方法 (検証用データを渡す)
```

<br>

# メソッド一覧
上記クラスは、以下のようなメソッドを持ちます

|メソッド名|機能|
|---|---|
|[plot_first_validation_curve]()|範囲を定めて検証曲線をプロットし、パラメータ調整範囲の参考とします|
|[grid_search_tuning]()|グリッドサーチを実行します|
|[random_search_tuning]()|ランダムサーチを実行します|
|[bayes_opt_tuning]()|BayesianOptimizationでベイズ最適化を実行します|
|[optuna_tuning]()|Optunaでベイズ最適化を実行します|
|[plot_search_history]()|チューニング進行に伴うスコアの上昇履歴をグラフ表示します|
|[get_search_history]()|チューニング進行に伴うスコアの上昇履歴をpandas.DataFrameで取得します|
|[plot_search_map]()|チューニング探索点と評価指標をマップ表示します (グリッドサーチはヒートマップ、それ以外は散布図)|
|[plot_best_learning_curve]()|学習曲線をプロットし、チューニング後モデルのバイアスとバリアンスの判断材料とします|
|[plot_best_validation_curve]()|学習後の検証曲線をプロットし、チューニング妥当性の判断材料とします|
|[plot_param_importances]()|パラメータを説明変数としてスコアをランダムフォレスト回帰した際のfeature_importancesを取得します。スコアの変化に寄与するパラメータの判断材料とします|
|[get_feature_importances]()|チューニング後の最適モデルのfeature_importancesを取得します (feature_importances対応学習器のみ)|
|[plot_feature_importances]()|チューニング後の最適モデルのfeature_importancesをグラフ表示します (feature_importances対応学習器のみ)|

※大半のメソッドは全てのクラスに対応していますが、
get_feature_importancesおよびplot_feature_importancesメソッドは、XGBoostおよびLightGBMのみ対応しています。

## plot_first_validation_curveメソッド
範囲を定めて検証曲線をプロットし、パラメータ調整範囲の参考とします

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なるESTIMATOR定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|最適化対象の学習器インスタンス。`not_opt_params`で指定したパラメータは上書きされるので注意|
|validation_<br>curve_params|オプション　　|dict[str, list[float]]|[クラスごとに異なるVALIDATION_CURVE_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|検証曲線プロット対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|not_opt_params|オプション|dict|[クラスごとに異なるNOT_OPT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`validation_curve_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なるPARAM_SCALES定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`validation_curve_params`のパラメータごとのスケール('linear', 'log')|
|plot_stats|オプション|str|'mean'|検証曲線グラフにプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))|
|axes|オプション|list[matplotlib.axes.Axes]|None|グラフ描画に使用するaxes (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|fit_params|オプション|dict|[クラスごとに異なるFIT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_first_validation_curve.py)
#### オプション引数指定なしで検証曲線プロット
オプション引数を指定しないとき、[前述のデフォルト値]()を使用してプロットします

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
###### デフォルト引数で検証曲線プロット ######
tuning.plot_first_validation_curve()
```
実行結果

<img width="800px" src="https://user-images.githubusercontent.com/59557625/145722194-0791ecc7-6fad-4c7a-a02e-71ec82d0c6bd.png">

#### パラメータ範囲を指定して検証曲線プロット
`validation_curve_params`引数で、検証曲線のパラメータ範囲を指定する事ができます

```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# データセット読込
TARGET_VARIABLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数多いので1000点にサンプリング
y = california_housing[TARGET_VARIABLE].values 
X = california_housing[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# パラメータ
VALIDATION_CURVE_PARAMS = {'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64],
                           'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
                           'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
                           'min_child_samples': [0, 5, 10, 20, 30, 50]
                           }
###### パラメータ範囲を指定して検証曲線プロット ######
tuning.plot_first_validation_curve(validation_curve_params=VALIDATION_CURVE_PARAMS)
```
実行結果

<img width="800px" src="https://user-images.githubusercontent.com/59557625/145722445-033a1ccb-1b71-45b7-a290-2b811620b754.png">

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L123)をご参照ください

<br>
<br>

## grid_search_tuningメソッド
グリッドサーチを実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なるESTIMATOR定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|最適化対象の学習器インスタンス。`not_opt_params`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なるTUNING_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error' in regression.'neg_log_loss' in clasification|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なるNOT_OPT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なるPARAM_SCALES定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_logging|オプション|{'inside','outside',None}|None|MLflowでの結果記録有無('inside':with構文で記録, 'outside':外部でRun実行, None:MLflow実行なし)。詳細は[こちら]()|
|mlflow_<br>tracking_uri|オプション|str|None|MLflowのTracking URI。[こちらを参照ください]()|
|mlflow_<br>artifact_location|オプション　　|str|None|MLflowのArtifact URI。[こちらを参照ください]()|
|mlflow_<br>experiment_name|オプション|str|None|MLflowのExperiment名。[こちらを参照ください]()|
|grid_kws|オプション|dict|None|sklearn.model_selection.GridSearchCVに渡す引数 (estimator, tuning_params, cv, scoring以外)|
|fit_params|オプション|dict|[クラスごとに異なるFIT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/grid_search_tuning.py)
#### オプション引数指定なしでグリッドサーチ
オプション引数を指定しないとき、[デフォルトの引数]()を使用してプロットします

```python
from param_tuning import RFRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でグリッドサーチ ######
best_params, best_score = tuning.grid_search_tuning()
```
実行結果

```
score before tuning = -0.018627075795771445
best_params = {'max_depth': 8, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 80}
score after tuning = -0.018313930236533316
```

#### パラメータ探索範囲を指定してグリッドサーチ
`tuning_params`引数で、グリッドサーチのパラメータ探索範囲を指定する事ができます

```python
from param_tuning import RFRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# パラメータ
CV_PARAMS_GRID = {'n_estimators': [20, 80, 160],
                  'max_depth': [2, 8, 32],
                  'min_samples_split': [2, 8, 32],
                  'min_samples_leaf': [1, 4, 16]
                  }
###### パラメータ範囲を指定してグリッドサーチ ######
best_params, best_score = tuning.grid_search_tuning(tuning_params=CV_PARAMS_GRID)
```
実行結果

```
score before tuning = -0.018627075795771445
best_params = {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 80}
score after tuning = -0.018313930236533316
```

#### 学習器を指定してグリッドサーチ
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です

```python
from param_tuning import RFRegressorTuning
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# 学習器を指定
ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
# パラメータ
CV_PARAMS_GRID = {'n_estimators': [20, 80, 160],
                  'max_features': [2, 5],
                  'max_depth': [2, 8, 32],
                  'min_samples_split': [2, 8, 32],
                  'min_samples_leaf': [1, 4, 16]
                  }
###### 学習器とパラメータ範囲を指定してグリッドサーチ ######
best_params, best_score = tuning.grid_search_tuning(estimator=ESTIMATOR,
                                                    tuning_params=CV_PARAMS_GRID)
```
実行結果

```
score before tuning = -0.01862916391210388
best_params = {'rf__max_depth': 8, 'rf__max_features': 2, 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 80}
score after tuning = -0.018483563545478098
```
※本来パイプラインのパラメータ名は`学習器名__パラメータ名`と指定する必要がありますが、本ツールの`tuning_params`には自動で学習器名を付加する機能を追加しているので、`パラメータ名`のみでも指定可能です (`fit_params`指定時も同様)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L128)をご参照ください

<br>
<br>

## random_search_tuningメソッド
ランダムサーチを実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なるESTIMATOR定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|最適化対象の学習器インスタンス。`not_opt_params`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なるTUNING_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|n_iter|オプション|int|[クラスごとに異なるN_ITER定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|ランダムサーチの試行数|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なるNOT_OPT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なるPARAM_SCALES定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_logging|オプション|{'inside','outside',None}|None|MLflowでの結果記録有無('inside':with構文で記録, 'outside':外部でRun実行, None:MLflow実行なし)。詳細は[こちら]()|
|mlflow_<br>tracking_uri|オプション|str|None|MLflowのTracking URI。[こちらを参照ください]()|
|mlflow_<br>artifact_location|オプション　　|str|None|MLflowのArtifact URI。[こちらを参照ください]()|
|mlflow_<br>experiment_name|オプション|str|None|MLflowのExperiment名。[こちらを参照ください]()|
|rand_kws|オプション|dict|None|sklearn.model_selection.RondomizedSearchCVに渡す引数 (estimator, tuning_params, cv, scoring, n_iter以外)|
|fit_params|オプション|dict|[クラスごとに異なるFIT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/random_search_tuning.py)
#### オプション引数指定なしでランダムサーチ
オプション引数を指定しないとき、[デフォルトの引数]()を使用してプロットします

```python
from param_tuning import RFRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でランダムサーチ ######
best_params, best_score = tuning.random_search_tuning()
```
実行結果

```
score before tuning = -0.018627075795771445
best_params = {'n_estimators': 60, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 2, 'max_depth': 8}
score after tuning = -0.017934841860748053
```

#### パラメータ探索範囲と試行数を指定してランダムサーチ
`tuning_params`引数で、ランダムサーチのパラメータ探索範囲を指定する事ができます。

また、`n_iter`引数で探索の試行数を指定できます

```python
from param_tuning import RFRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# パラメータ
CV_PARAMS_RANDOM = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                    'max_features': [2, 3, 4, 5],
                    'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                    }
###### パラメータ範囲と試行数を指定してランダムサーチ ######
best_params, best_score = tuning.random_search_tuning(tuning_params=CV_PARAMS_RANDOM,
                                                      n_iter=160)
```
実行結果
```
score before tuning = -0.018627075795771445
best_params = {'n_estimators': 40, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 4, 'max_depth': 4}
score after tuning = -0.01786570144420851
```

#### 学習器を指定してランダムサーチ
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です

```python
from param_tuning import RFRegressorTuning
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# 学習器を指定
ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
# パラメータ
CV_PARAMS_RANDOM = {'n_estimators': [20, 30, 40, 60, 80, 120, 160],
                    'max_features': [2, 3, 4, 5],
                    'max_depth': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_split': [2, 3, 4, 6, 8, 12, 16, 24, 32],
                    'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16]
                    }
###### 学習器を指定してランダムサーチ ######
best_params, best_score = tuning.random_search_tuning(estimator=ESTIMATOR,
                                                      tuning_params=CV_PARAMS_RANDOM,
                                                      n_iter=160)
```
実行結果

```
score before tuning = -0.01862916391210388
best_params = {'rf__n_estimators': 40, 'rf__min_samples_split': 4, 'rf__min_samples_leaf': 1, 'rf__max_features': 4, 'rf__max_depth': 4}
score after tuning = -0.01786570144420851
```
※本来パイプラインのパラメータ名は`学習器名__パラメータ名`と指定する必要がありますが、本ツールの`tuning_params`には自動で学習器名を付加する機能を追加しているので、`パラメータ名`のみでも指定可能です (`fit_params`指定時も同様)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_svr_regression.py#L155)をご参照ください

<br>
<br>

## bayes_opt_tuningメソッド
[BayesianOptimizationライブラリ](https://github.com/fmfn/BayesianOptimization)によるベイズ最適化を実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なるESTIMATOR定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|最適化対象の学習器インスタンス。`not_opt_params`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なるTUNING_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (BayesianOptimization初期化時の`random_state`引数、および学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|n_iter|オプション|int|[クラスごとに異なるN_ITER定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|ベイズ最適化の試行数|
|init_points|オプション|int|[クラスごとに異なるINIT_POINTS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|ランダムな初期探索点の個数|
|acq|オプション|{'ei', 'pi', 'ucb'}|'ei'|獲得関数 ('ei': EI戦略, 'pi': PI戦略, 'ucb': UCB戦略)|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なるNOT_OPT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`以外のチューニング対象外パラメータを指定|
|int_params|オプション|int|[クラスごとに異なるINT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|整数型のパラメータ一覧のリスト|
|param_scales|オプション|dict[str, str]|[クラスごとに異なるPARAM_SCALES定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_logging|オプション|{'inside','outside',None}|None|MLflowでの結果記録有無('inside':with構文で記録, 'outside':外部でRun実行, None:MLflow実行なし)。詳細は[こちら]()|
|mlflow_<br>tracking_uri|オプション|str|None|MLflowのTracking URI。[こちらを参照ください]()|
|mlflow_<br>artifact_location|オプション　　|str|None|MLflowのArtifact URI。[こちらを参照ください]()|
|mlflow_<br>experiment_name|オプション|str|None|MLflowのExperiment名。[こちらを参照ください]()|
|fit_params|オプション|dict|[クラスごとに異なるFIT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/bayes_opt_tuning.py)

#### オプション引数指定なしでBayesianOptimization
オプション引数を指定しないとき、[デフォルトの引数]()を使用してBayesianOptimizationでチューニングします

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でBayesianOptimization ######
best_params, best_score = tuning.bayes_opt_tuning()
```
実行結果

```
score before tuning = -0.03336768277100166

|   iter    |  target   | colsam... | min_ch... | num_le... | reg_alpha | reg_la... | subsample | subsam... |
-------------------------------------------------------------------------------------------------------------
|  1        | -0.03337  |  0.6247   |  47.54    |  37.14    | -2.204    | -3.532    |  0.4936   |  0.4066   |
|  2        | -0.03337  |  0.9197   |  30.06    |  35.99    | -3.938    | -1.09     |  0.8995   |  1.486    |
  :
  :
|  70       | -0.01939  |  0.9147   |  3.103    |  39.95    | -3.717    | -3.67     |  0.512    |  2.78     |
=============================================================================================================
best_params = {'colsample_bytree': 0.7157900674004883, 'min_child_samples': 6, 'num_leaves': 42, 'reg_alpha': 0.0024676255164280147, 'reg_lambda': 0.0025350316484909954, 'subsample': 0.8763786058923628, 'subsample_freq': 1}
score after tuning = -0.015674379678829172
```

#### パラメータ範囲と試行数を指定してBayesianOptimization
`tuning_params`引数で、ベイズ最適化のパラメータ探索範囲を指定する事ができます。

また、`n_iter`引数でベイズ最適化の試行数を指定できます

探索の合計試行数は、`init_points`で指定したランダム初期点数 + `n_iter`となります

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# パラメータ
BAYES_PARAMS = {'reg_alpha': (0.001, 0.1),
                'reg_lambda': (0.001, 0.1),
                'num_leaves': (2, 50),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.4, 0.8),
                'subsample_freq': (0, 5),
                'min_child_samples': (0, 20)
                }
###### パラメータ範囲と試行数を指定してBayesianOptimization ######
best_params, best_score = tuning.bayes_opt_tuning(tuning_params=CV_PARAMS_RANDOM,
                                                  n_iter=160,
                                                  init_points=10)
```
実行結果

```
score before tuning = -11.979161807916636

|   iter    |  target   | colsam... | min_ch... | num_le... | reg_alpha | reg_la... | subsample | subsam... |
-------------------------------------------------------------------------------------------------------------
|  1        | -0.03337  |  0.6247   |  19.01    |  37.14    | -1.803    | -2.688    |  0.4624   |  0.2904   |
|  2        | -0.03337  |  0.9197   |  12.02    |  35.99    | -2.959    | -1.06     |  0.733    |  1.062    |
  :
  :
|  85       | -0.02821  |  1.0      |  0.0      |  22.21    | -1.0      | -1.0      |  0.4      |  5.0      |
=============================================================================================================
best_params = {'colsample_bytree': 0.7663273032020579, 'min_child_samples': 5, 'num_leaves': 50, 'reg_alpha': 0.004886609538667352, 'reg_lambda': 0.0012229829134962934, 'subsample': 0.79020445435527, 'subsample_freq': 4}
score after tuning = -0.016980186157125283
```

#### 学習器を指定してBayesianOptimization
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# 学習器を指定
ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("rf", LGBMRegressor())])
# パラメータ
BAYES_PARAMS = {'reg_alpha': (0.001, 0.1),
                'reg_lambda': (0.001, 0.1),
                'num_leaves': (2, 50),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.4, 0.8),
                'subsample_freq': (0, 5),
                'min_child_samples': (0, 20)
                }
###### 学習器を指定してBayesianOptimization ######
best_params, best_score = tuning.bayes_opt_tuning(estimator=ESTIMATOR,
                                                  tuning_params=BAYES_PARAMS,
                                                  n_iter=75,
                                                  init_points=10)
```
実行結果

```
score before tuning = -0.03336768277100166
|   iter    |  target   | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... |
-------------------------------------------------------------------------------------------------------------
|  1        | -0.03337  |  0.6247   |  19.01    |  37.14    | -1.803    | -2.688    |  0.4624   |  0.2904   |
|  2        | -0.03337  |  0.9197   |  12.02    |  35.99    | -2.959    | -1.06     |  0.733    |  1.062    |
  :
  :
|  85       | -0.01771  |  1.0      |  3.12     |  14.31    | -1.803    | -2.383    |  0.8      |  4.253    |
=============================================================================================================
best_params = {'lgbmr__colsample_bytree': 0.8712626692669376, 'lgbmr__min_child_samples': 5, 'lgbmr__num_leaves': 15, 'lgbmr__reg_alpha': 0.0013795818196214403, 'lgbmr__reg_lambda': 0.06566397910906291, 'lgbmr__subsample': 0.7544905867317223, 'lgbmr__subsample_freq': 5}
score after tuning = -0.016169025610778674
```
※本来パイプラインのパラメータ名は`学習器名__パラメータ名`と指定する必要がありますが、本ツールの`tuning_params`には自動で学習器名を付加する機能を追加しているので、`パラメータ名`のみでも指定可能です (`fit_params`指定時も同様)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_svr_regression.py#L202)をご参照ください

<br>
<br>

## optuna_tuningメソッド
[Optunaライブラリ](https://www.preferred.jp/ja/projects/optuna/)によるベイズ最適化を実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なるESTIMATOR定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|最適化対象の学習器インスタンス。`not_opt_params`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なるTUNING_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (TPESamplerの`seed`引数、および学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|n_trials|オプション|int|[クラスごとに異なるN_TRIALS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|ベイズ最適化の試行数|
|study_kws|オプション|dict|{'sampler': TPESampler(), 'direction': 'maximize'}|optuna.study.create_study()に渡す引数|
|optimize_kws|オプション|dict|{}|optuna.study.Study.optimize()に渡す引数 (n_trials以外)|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なるNOT_OPT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`以外のチューニング対象外パラメータを指定|
|int_params|オプション|int|[クラスごとに異なるINT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|整数型のパラメータ一覧のリスト|
|param_scales|オプション|dict[str, str]|[クラスごとに異なるPARAM_SCALES定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_logging|オプション|{'inside','outside',None}|None|MLflowでの結果記録有無('inside':with構文で記録, 'outside':外部でRun実行, None:MLflow実行なし)。詳細は[こちら]()|
|mlflow_<br>tracking_uri|オプション|str|None|MLflowのTracking URI。[こちらを参照ください]()|
|mlflow_<br>artifact_location|オプション　　|str|None|MLflowのArtifact URI。[こちらを参照ください]()|
|mlflow_<br>experiment_name|オプション|str|None|MLflowのExperiment名。[こちらを参照ください]()|
|fit_params|オプション|dict|[クラスごとに異なるFIT_PARAMS定数](https://c60evaporator.github.io/muscle-tuning/each_estimators.html)|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/optuna_tuning.py)
#### オプション引数指定なしでOptunaチューニング
オプション引数を指定しないとき、[デフォルトの引数]()を使用してOptunaでチューニングします
```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でOptunaチューニング ######
best_params, best_score = tuning.optuna_tuning()
```
実行結果

```
[I 2021-12-14 00:30:14,655] A new study created in memory with name: no-name-2909a142-5e22-40de-b896-5c7e00f7fbe2
score before tuning = -0.03336768277100166

[I 2021-12-14 00:30:15,077] Trial 0 finished with value: -0.01770816915416739 and parameters: {'reg_alpha': 0.0013292918943162175, 'reg_lambda': 0.07114476009343425, 'num_leaves': 37, 'colsample_bytree': 0.759195090518222, 'subsample': 0.4936111842654619, 'subsample_freq': 1, 'min_child_samples': 2}. Best is trial 0 with value: -0.01770816915416739.
  :
  :
[I 2021-12-14 00:31:10,918] Trial 199 finished with value: -0.018916315858869295 and parameters: {'reg_alpha': 0.00036956617235105757, 'reg_lambda': 0.08791527491442741, 'num_leaves': 48, 'colsample_bytree': 0.792614835286705, 'subsample': 0.9907365711754348, 'subsample_freq': 0, 'min_child_samples': 4}. Best is trial 106 with value: -0.015348697968670694.
best_params = {'reg_alpha': 0.0006347418019359087, 'reg_lambda': 0.059722944467666184, 'num_leaves': 47, 'colsample_bytree': 0.7768227979321171, 'subsample': 0.9358047309028262, 'subsample_freq': 3, 'min_child_samples': 6}
score after tuning = -0.015348697968670694
```

#### パラメータ範囲と試行数を指定してOptunaチューニング実行
`tuning_params`引数で、ベイズ最適化のパラメータ探索範囲を指定する事ができます。

また、`n_trials`引数でベイズ最適化の試行数を指定できます

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# パラメータ
BAYES_PARAMS = {'reg_alpha': (0.001, 0.1),
                'reg_lambda': (0.001, 0.1),
                'num_leaves': (2, 50),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.4, 0.8),
                'subsample_freq': (0, 5),
                'min_child_samples': (0, 20)
                }
###### パラメータ範囲と試行数を指定してOptunaチューニング ######
best_params, best_score = tuning.optuna_tuning(tuning_params=BAYES_PARAMS,
                                               n_trials=200,
                                               )
```
実行結果

```
[I 2021-12-14 00:38:27,069] A new study created in memory with name: no-name-8b1a53be-678f-479f-a113-b166292b030b
score before tuning = -0.03336768277100166

[I 2021-12-14 00:38:27,564] Trial 0 finished with value: -0.019595974792610084 and parameters: {'reg_alpha': 0.005611516415334507, 'reg_lambda': 0.07969454818643935, 'num_leaves': 37, 'colsample_bytree': 0.759195090518222, 'subsample': 0.46240745617697465, 'subsample_freq': 0, 'min_child_samples': 1}. Best is trial 0 with value: -0.019595974792610084.

  :
  :
[I 2021-12-14 00:39:18,042] Trial 199 finished with value: -0.016574941047789542 and parameters: {'reg_alpha': 0.002123392288572352, 'reg_lambda': 0.02246543428565874, 'num_leaves': 30, 'colsample_bytree': 0.7186823509475074, 'subsample': 0.7025863258246245, 'subsample_freq': 2, 'min_child_samples': 5}. Best is trial 128 with value: -0.016369607641459265.
best_params = {'reg_alpha': 0.0010080175418796056, 'reg_lambda': 0.017442918126437838, 'num_leaves': 22, 'colsample_bytree': 0.8349338860971882, 'subsample': 0.7281796185085103, 'subsample_freq': 2, 'min_child_samples': 4}
score after tuning = -0.016369607641459265
```

#### 学習器を指定してOptunaチューニング実行
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です
```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
# 学習器を指定
ESTIMATOR = Pipeline([("scaler", StandardScaler()), ("lgbmr", LGBMRegressor())])
# パラメータ
BAYES_PARAMS = {'reg_alpha': (0.001, 0.1),
                'reg_lambda': (0.001, 0.1),
                'num_leaves': (2, 50),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.4, 0.8),
                'subsample_freq': (0, 5),
                'min_child_samples': (0, 20)
                }
###### 学習器を指定してOptunaチューニング ######
best_params, best_score = tuning.optuna_tuning(estimator=ESTIMATOR,
                                               tuning_params=BAYES_PARAMS,
                                               n_trials=200)
```
実行結果

```
[I 2021-12-14 00:44:27,750] A new study created in memory with name: no-name-9c9a5ce9-78fc-4d58-88b1-28e1e62f57d6
score before tuning = -0.03336768277100166

[I 2021-12-14 00:44:28,261] Trial 0 finished with value: -0.019134840886528483 and parameters: {'lgbmr__reg_alpha': 0.005611516415334507, 'lgbmr__reg_lambda': 0.07969454818643935, 'lgbmr__num_leaves': 37, 'lgbmr__colsample_bytree': 0.759195090518222, 'lgbmr__subsample': 0.46240745617697465, 'lgbmr__subsample_freq': 0, 'lgbmr__min_child_samples': 1}. Best is trial 0 with value: -0.019134840886528483.
  :
  :
[I 2021-12-14 00:45:26,385] Trial 149 finished with value: -0.020414543680491087 and parameters: {'lgbmr__reg_alpha': 0.0018662271931017159, 'lgbmr__reg_lambda': 0.006697621112757029, 'lgbmr__num_leaves': 22, 'lgbmr__colsample_bytree': 0.6310482276089768, 'lgbmr__subsample': 0.6348418804738211, 'lgbmr__subsample_freq': 0, 'lgbmr__min_child_samples': 8}. Best is trial 132 with value: -0.014828721902099879.
best_params = {'lgbmr__reg_alpha': 0.0026605704978420385, 'lgbmr__reg_lambda': 0.005932725865394271, 'lgbmr__num_leaves': 18, 'lgbmr__colsample_bytree': 0.7166351013315516, 'lgbmr__subsample': 0.5599413528050566, 'lgbmr__subsample_freq': 0, 'lgbmr__min_child_samples': 6}
score after tuning = -0.014828721902099879
```
※本来パイプラインのパラメータ名は`学習器名__パラメータ名`と指定する必要がありますが、本ツールの`tuning_params`には自動で学習器名を付加する機能を追加しているので、`パラメータ名`のみでも指定可能です (`fit_params`指定時も同様)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L199)をご参照ください

<br>
<br>


## plot_search_historyメソッド
チューニング進行に伴うスコアの上昇履歴をグラフ表示します

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|x_axis|オプション|{'index', 'time'}|'index'|横軸の種類 ('index':試行回数, 'time':経過時間(合計時間での補正値))|
|plot_kws|オプション|dist|None|プロット用のmatplotlib.pyplot.plotに渡す引数|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_search_history.py)
#### オプション引数指定なしでスコアの上昇履歴を表示
オプション引数を指定しないとき、[デフォルトの引数]()を使用してOptunaでチューニングします
```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でスコアの上昇履歴を表示 ######
tuning.plot_search_history()
```
実行結果

![history](https://user-images.githubusercontent.com/59557625/145845317-ed71e1e7-35f7-474a-bdac-cdfefa4be659.png)

#### 横軸に時間を指定してスコアの上昇履歴を表示
`x_axis`引数='time'と指定する事で、横軸を試行数 → 時間に変更する事ができます。

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### 時間を横軸に指定してスコアの上昇履歴を表示 ######
tuning.plot_search_history(x_axis='time')
```
実行結果

![history_time](https://user-images.githubusercontent.com/59557625/145845648-e3424bb6-8fb9-40b7-bc74-af9971c43c45.png)

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L135)をご参照ください

<br>
<br>

## get_search_historyメソッド
チューニング進行に伴うスコアの上昇履歴 (plot_search_historyと同内容)をPandas DataFrameで取得します。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

引数はありません

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/get_search_history.py)

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でスコアの上昇履歴を取得 ######
df_history = tuning.get_search_history()
df_history
```
実行結果

```
reg_alpha	reg_lambda	num_leaves	colsample_bytree	subsample	subsample_freq	min_child_samples	test_score	raw_trial_time	max_score	raw_total_time	total_time
0	0.001329	0.071145	37	0.759195	0.493611	1	2	-0.017708	0.429852	-0.017708	0.429852	0.431247
...	...	...	
199	0.000370	0.087915	48	0.792615	0.990737	0	4	-0.018916	0.314449	-0.015349	55.788442	55.969547
```

<br>
<br>

## plot_search_mapメソッド
探索履歴 (グリッドサーチ：ヒートマップ、その他：散布図)をプロットします。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|order|オプション|list[str]|None|軸のパラメータ名を指定。Noneならparam_importances順に自動指定|
|pair_n|オプション|int|4|図を並べる枚数 (グリッドサーチ以外)|
|rounddigits_title|オプション|int|3|グラフタイトルのパラメータ値の丸め桁数 (グリッドサーチ以外)|
|rank_number|オプション|int|None|スコア上位何番目までを文字表示するか。Noneなら表示なし|
|rounddigits_score|オプション|int|3|上位スコア表示の丸め桁数|
|subplot_kws|オプション|dict|None|プロット用のplt.subplots()に渡す引数 (例：figsize)|
|heat_kws|オプション|dict|None|ヒートマップ用のsns.heatmap()に渡す引数 (グリッドサーチのみ)|
|scatter_kws|オプション|dict|None|プロット用のplt.subplots()に渡す引数 (グリッドサーチ以外)|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_search_map.py)
#### オプション引数指定なしで探索履歴をプロット
オプション引数を指定しないとき、[デフォルトの引数]()を使用して探索履歴をプロットします
```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でスコアの上昇履歴を表示 ######
tuning.plot_search_map()
```
実行結果

<img width="720px" src="https://user-images.githubusercontent.com/59557625/146015194-811bf37c-d487-4ade-ab06-b57cb5f34c6d.png">

#### 図の枚数と軸のパラメータを指定してスコアの上昇履歴を表示
`pair_n`引数で、プロットする図の縦横枚数を指定する事ができます

`order`引数で、軸のパラメータをリストで指定する事ができます。リスト指定順に、グラフ横軸 → グラフ縦軸 → 全体縦軸 → 全体横軸の順番でプロットされます。`order`=Noneなら[param_importances]()順にプロットされます。

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### 時間を横軸に指定してスコアの上昇履歴を表示 ######
tuning.plot_search_map(pair_n=6, 
                       order=['min_child_samples', 'colsample_bytree', 'subsample', 'num_leaves'])
```
実行結果

<img width="840px" src="https://user-images.githubusercontent.com/59557625/146015925-c9c571cc-23a0-4f3f-bb12-2792073a0ee3.png">

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L211)をご参照ください

<br>
<br>

## plot_best_learning_curveメソッド
チューニング後の学習曲線をプロットします。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|plot_stats|オプション|{'mean', 'median'}|'mean'|学習曲線としてプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_best_learning_curve.py)
#### オプション引数指定なしで学習曲線をプロット
オプション引数を指定しないとき、[デフォルトの引数]()を使用して学習曲線をプロットします
```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数で学習曲線を表示 ######
tuning.plot_best_learning_curve()
```
実行結果

![learningcurve](https://user-images.githubusercontent.com/59557625/146016500-0c141127-ab95-4f72-a43e-5c4418312caf.png)

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L215)をご参照ください

<br>
<br>

## plot_best_validation_curveメソッド
チューニング後の学習曲線をプロットします。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|validation_<br>curve_params|オプション　　|dict[str, list[float]]|[クラスごとに異なる]()|検証曲線プロット対象のパラメータ範囲|
|param_scales|オプション|dict[str, str]|[クラスごとに異なる]()|`validation_curve_params`のパラメータごとのスケール('linear', 'log')|
|plot_stats|オプション|{'mean', 'median'}|'mean'|学習曲線としてプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))|
|axes|オプション|list[matplotlib.axes.Axes]|None|グラフ描画に使用するaxes (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_best_validation_curve.py)
#### オプション引数指定なしで学習曲線をプロット
オプション引数を指定しないとき、[デフォルトの引数]()を使用して検証曲線をプロットします

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数で検証曲線を表示 ######
tuning.plot_best_validation_curve()
```
実行結果

<img width="800px" src="https://user-images.githubusercontent.com/59557625/146039412-3a0f99a8-de96-480a-9502-941dbd6a96b1.png">

#### パラメータ範囲を指定して検証曲線プロット
`validation_curve_params`引数で、検証曲線のパラメータ範囲を指定する事ができます

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
# パラメータ
VALIDATION_CURVE_PARAMS = {'reg_lambda': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64],
                           'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
                           'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
                           'min_child_samples': [0, 5, 10, 20, 30, 50]
                           }
###### パラメータ範囲を指定して検証曲線プロット ######
tuning.plot_best_validation_curve(validation_curve_params=VALIDATION_CURVE_PARAMS)
```
実行結果

<img width="800px" src="https://user-images.githubusercontent.com/59557625/146040510-90bf88a7-9854-43d5-abcb-edb2fde26203.png">

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L217)をご参照ください

<br>
<br>

## plot_param_importancesメソッド
パラメータを説明変数、評価指標を目的変数としてランダムフォレスト回帰を行い、求めたfeature_importancesを**param_importances**としてプロットします。

**評価指標向上に寄与したパラメータ**の判断に使用することができます（チューニングしたパラメータ範囲に大きく依存するので、あくまで目安として使用してください）

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれか、およびplot_search_map()を実行する必要があります

引数はありません

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_param_importances.py)

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
tuning.plot_search_map()  # plot_search_map実行
###### param_importancesを表示 ######
tuning.plot_param_importances()
```
実行結果

![param_importance](https://user-images.githubusercontent.com/59557625/146016932-d970d6ab-0b23-4927-96e9-a0aca0f9eca1.png)

<br>
<br>

## plot_feature_importancesメソッド
チューニング後の学習器のfeature_importancesをプロットします。

feature_importances_算出に対応した学習器(ランダムフォレスト、LightGBM、XGBooost)のみプロット可能です。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|ax|オプション|matplotlib.axes.Axes|None|表示対象のax (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/plot_feature_importances.py)
#### オプション引数指定なしでfeature_importancesをプロット
オプション引数を指定しないとき、[デフォルトの引数]()を使用して学習曲線をプロットします

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でfeature_importancesを表示 ######
tuning.plot_feature_importances()
```
実行結果

![featureimportance](https://user-images.githubusercontent.com/59557625/146017482-a5c5b422-7b42-4edd-817e-ab0faeb11d4b.png)

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L221)をご参照ください

<br>
<br>

## get_feature_importancesメソッド
チューニング後の学習器のfeature_importances (plot_feature_importancesと同内容)をPandasのDataFrameで取得します

feature_importances_算出に対応した学習器(ランダムフォレスト、LightGBM、XGBooost)のみプロット可能です。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

引数はありません

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/blob/master/examples/method_examples/get_feature_importances.py)
#### feature_importancesを取得
オプション引数を指定しないとき、[デフォルトの引数]()を使用して学習曲線をプロットします

```python
from param_tuning import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### チューニング後feature_importancesを取得 ######
importances = tuning.get_feature_importances()
importances
```
実行結果

```
      feature_name       importance
0	latitude           1310
1	5_household_member 896
2	3_male_ratio       1338
3	2_between_30to60   1061
```

# プロパティ一覧

|プロパティ名|型|概要|
|---|---|---|
|X|numpy.ndarray|説明変数データ|
|y|numpy.ndarray|説明変数データ|
|x_colnames|list[str]|説明変数の名称|
|y_colname|list[str]|目的変数の名称|
|tuning_params|dict[str, {list, tuple}]|チューニング対象のパラメータ一覧|
|not_opt_params|dict[str]|チューニング対象外のパラメータ一覧|
|int_params|list[str]|整数型のパラメータのリスト(ベイズ最適化のみ)|
|param_scales|dict[str, {'linear', 'log'}]|パラメータのスケール|
|scoring|str|チューニングで最大化する評価スコア|
|seed|int|乱数シード(クロスバリデーション分割、ベイズ最適化のサンプラー等で使用)|
|cv|cross-validation generator|クロスバリデーション分割法|
|estimator|estimator object implementing 'fit'|チューニング対象の学習器インスタンス|
|learner_name|str|パイプライン処理時の学習器名称(estimatorがパイプラインのときのみ)|
|fit_params|str|学習時のfit()に渡すパラメータをdict指定|
|score_before|float|チューニング前のスコア|
|tuning_algo|{'grid', 'random', 'bayes-opt', 'optuna'}|チューニングに使用したアルゴリズム名|
|best_params|dict[str, {float, int}]|チューニング後の最適パラメータ|
|best_score|float|チューニング後のスコア|
|elapsed_time|float|所要時間|
|best_estimator|estimator object implementing 'fit'|チューニング後の学習モデル|
|search_history|dict[str]|探索履歴(パラメータ名をキーとしたdict)|
|param_importances|pandas.Series|ランダムフォレストで求めた各パラメータのスコアに対する重要度(x_colnamesの順)|

# 定数一覧
各機械学習アルゴリズムごとのデフォルト値は、[クラス一覧](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/api_each.md#クラス一覧)項の、「デフォルトパラメータのリンク」のリンク先を参照ください

|プロパティ名|型|概要|
|---|---|---|
|SEED|int|各チューニング実行メソッドの引数`seed`のデフォルト値 (乱数シード)|
|CV_NUM|int|各チューニング実行メソッドの引数`cv`のデフォルト値 (クロスバリデーション分割数)|
|ESTIMATOR|int|各チューニング実行メソッドの引数`estimator`のデフォルト値 (チューニング対象の学習器インスタンス)|
