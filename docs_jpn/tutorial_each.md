# 詳細チューニング 実行手順
[学習器の種類に合わせてクラスを選択](https://github.com/c60evaporator/tune-easy/blob/master/README.md#02-チューニング用クラスの初期化)し、**下図の手順**([こちらの記事に詳細](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#2-%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%AE%E6%89%8B%E9%A0%86%E3%81%A8%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E4%B8%80%E8%A6%A7))でパラメータチューニングを実施します。

2022/1現在、Scikit-LearnのAPIに対応した学習器が対象となります。

<img width="600" src="https://user-images.githubusercontent.com/59557625/130362754-a85fc7fc-38f7-4d5a-9e8f-c24321a2ed98.png">

## 手順一覧
**0. チューニングの準備**

&nbsp;├─ [0.1. データの読込＆前処理](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#01-データの読込前処理)

&nbsp;└─ [0.2. チューニング用クラスの初期化](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#02-チューニング用クラスの初期化)

**1. 最大化したい評価指標を定義**

&nbsp;└─ [1.1. 評価指標の選択](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#1-評価指標の選択)

**2. パラメータ探索範囲の選択**

&nbsp;└─ [2.1. パラメータ探索範囲の選択](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#2-パラメータ探索範囲の選択)

**3. 探索法を選択**

&nbsp;└─ [3.1. 探索手法を選択](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#3-探索法を選択)

**4. クロスバリデーションでチューニングを実行**

&nbsp;├─ [4.1. クロスバリデーション手法を選択](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#41-クロスバリデーション手法を選択)

&nbsp;├─ [4.2. チューニング前のスコアを確認](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#42-チューニング前のスコアを確認)

&nbsp;└─ [4.3. チューニング実行](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#43-チューニング実行)

**5. 学習曲線・検証曲線等でチューニング結果を確認**

&nbsp;├─ [5.1. チューニング履歴の確認](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#51-チューニング履歴の確認)

&nbsp;├─ [5.2. パラメータと評価指標の関係をマップ表示して確認](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#52-パラメータと評価指標の関係をマップ表示して確認)

&nbsp;├─ [5.3. 学習曲線を確認](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#53-学習曲線を確認)

&nbsp;└─ [5.4. 検証曲線を確認](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#54-検証曲線を確認)

**6. チューニング結果の活用**

&nbsp;└─ [6.1. チューニング後の学習器を使用する](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#6-チューニング後の学習器を使用する)

<br>
<br>

**App. MLflowによる結果ロギング**
[MLflowによる結果ロギング]()

<br>

## 詳細手順
### 0.1. データの読込＆前処理
使用するデータを読み込み、特徴量選択等の前処理を実行します。

#### 実行例
カリフォルニア住宅価格データセットを読み込み、特徴量 (説明変数)を選択します

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# データセット読込
TARGET_VARIABLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 選択した4説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数多いので1000点にサンプリング
y = california_housing[TARGET_VARIALBLE].values
X = california_housing[USE_EXPLANATORY].values
```

※選択した4特徴量は、ランダムフォレストのRFE (再帰的特徴量削減)により選定

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
# データセット読込
TARGET_VARIABLE = 'price'  # 目的変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)
y = california_housing[TARGET_VARIABLE].values
X_all = california_housing[fetch_california_housing().feature_names].values  # 全特徴量
# RFEで特徴量選択
selector = RFE(RandomForestRegressor(random_state=42), n_features_to_select=5)
selector.fit(X_all, y)
print(fetch_california_housing().feature_names)
print(selector.get_support())
```

```
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
[ True  True False False False  True  True  True]
```
特徴量選択については、[Scikit-Learn公式](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)を参照ください。

上例では、RFEで選ばれた5特徴量のうちLatitudeとLongitudeの相関係数が高い（0.9）ので、多重共線性防止のため後者を除外した4特徴量を採用しています。

### 0.2. チューニング用クラスの初期化
以下を参考に使用したい学習器に合わせてチューニング用クラスを選択し、インスタンスを作成します

|問題の種類|学習器の種類|クラス名|
|---|---|---|
|回帰|LightGBM回帰|[LGBMRegressorTuning]()|
|回帰|XGBoost回帰|[XGBRegressorTuning]()|
|回帰|サポートベクター回帰|[SVMRegressorTuning]()|
|回帰|ランダムフォレスト回帰|[RFRegressorTuning]()|
|回帰|ElasticNet|[ElasticNetTuning]()|
|分類|LightGBM分類|[LGBMClassifierTuning]()|
|分類|XGBoost分類|[XGBClassifierTuning]()|
|分類|サポートベクターマシン分類|[SVMClassifierTuning]()|
|分類|ランダムフォレスト分類|[RFClassifierTuning]()|
|分類|ロジスティック回帰|[LogisticRegressionTuning]()|
※他のアルゴリズムも追加希望あればIssuesに書き込んで下さい

#### 実行例
LightGBM回帰のチューニング用クラス初期化

```python
from param_tuning import LGBMRegressorTuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, eval_set_selection='all')
```
※eval_set_selection引数に関しては[こちらのリンク](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#-eval_data_sourceの指定値によるeval_setに入るデータの変化)参照

### 1. 評価指標の選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#21-%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99%E3%81%AE%E5%AE%9A%E7%BE%A9)チューニングの評価指標を選択します。

デフォルト(各メソッドの`scoring`引数を指定しないとき)では、以下の指標を使用します
|手法|デフォルトで使用する手法|文字列|
|---|---|---|
|回帰|RMSE|'neg_root_mean_squared_error'|
|2クラス分類|LogLoss|'neg_log_loss'|
|多クラス分類|LogLoss|'neg_log_loss'|

#### 実行例
MSEを指標に使用するとき

```
SCORING = 'neg_mean_squared_error'
```

### 2. パラメータ探索範囲の選択
[`plot_first_validation_curve()`]()メソッドで検証曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#22-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E7%A8%AE%E9%A1%9E%E3%81%A8%E6%8E%A2%E7%B4%A2%E7%AF%84%E5%9B%B2%E3%81%AE%E9%81%B8%E6%8A%9E)パラメータ探索範囲を選択します

事前に[4.1. クロスバリデーション手法の選択]()を実施し、`cv`引数に指定する事が望ましいです

#### 実行例
範囲を指定して検証曲線を描画

```python
VALIDATION_CURVE_PARAMS = {'reg_alpha': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                           'reg_lambda': [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10],
                           'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 192, 256],
                           'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
                           'min_child_samples': [0, 2, 5, 10, 20, 30, 50, 70, 100]
                           }
tuning.plot_first_validation_curve(validation_curve_params=VALIDATION_CURVE_PARAMS,
                                   scoring=SCORING,
                                   cv=KFold(n_splits=5, shuffle=True, random_state=42))
```
実行結果

<img width="800" src="https://user-images.githubusercontent.com/59557625/146211814-5ac00ce2-c2cf-4292-9e38-5036ea3c2669.png">

下図 (SVMでの実行例)のように検証曲線から過学習にも未学習にもなりすぎていない範囲を抽出し、探索範囲とすることが望ましいです

<img width="720" src="https://user-images.githubusercontent.com/59557625/130347923-c3ed17a2-8ad6-4f30-8ff7-fc91cf8f97ee.png">

### 3. 探索法を選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E6%89%8B%E9%A0%8634-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E9%81%B8%E6%8A%9E%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)パラメータの探索法を選択します。
以下の4種類の探索法から使用したい手法を選び、対応したメソッドを選択します

|探索法|使用するメソッド|
|---|---|
|グリッドサーチ|[grid_search_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#grid_search_tuningメソッド)|
|ランダムサーチ|[random_search_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)|
|ベイズ最適化 (BayesianOptimization)|[bayes_opt_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)|
|ベイズ最適化 (Optuna)|[optuna_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)|

### 4.1. クロスバリデーション手法を選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#24-%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)クロスバリデーションの手法を選択します。

#### 実行例
クロスバリデーションに5分割KFoldを指定

```python
CV = KFold(n_splits=5, shuffle=True, random_state=42)
```

### 4.2. チューニング前のスコアを確認
`※チューニング前スコアは4.3実行時にも表示されるので、本工程は飛ばしても構いません`

チューニング前のスコアを確認します。
このとき

・学習器の`fit()`メソッドに渡す引数（LightGBMの`early_stopping_rounds`等）

・チューニング対象外パラメータを学習器インスタンスに渡す事

を忘れないようにご注意ください

#### 実行例
LightGBM回帰において、`fit()`メソッドに渡す引数`fit_params`およびチューニング対象外パラメータ`not_opt_params`を指定してスコア算出

```python
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
# 学習器のfit()メソッドに渡す引数
FIT_PARAMS = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'rmse',
              'eval_set': [(X, y)]
              }
# チューニング対象外パラメータ
NOT_OPT_PARAMS = {'objective': 'regression',
                  'random_state': 42,
                  'boosting_type': 'gbdt',
                  'n_estimators': 10000
                  }
# 学習器のインスタンス作成
lgbmr = LGBMRegressor(**NOT_OPT_PARAMS)
# クロスバリデーションでスコア算出
scores = cross_val_score(lgbmr, X, y,
                         scoring=SCORING,  # 評価指標 (1で選択)
                         cv=CV,  # クロスバリデーション手法 (4.1で選択)
                         fit_params=FIT_PARAMS  # 学習器のfit()メソッド引数
                         )
print(np.mean(scores))
```
実行結果

```
-0.4561245619412457
```
<br>

[seaborn-analyzer](https://pypi.org/project/seaborn-analyzer/)ライブラリを活用して予測値と実測値の関係を可視化すると、挙動がわかりやすくなるのでお勧めです

```python
from seaborn_analyzer import regplot
california_housing['price'] = y
regplot.regression_pred_true(lgbmr,
                             x=tuning.x_colnames,
                             y='price',
                             data=california_housing,
                             scores='mse',
                             cv=CV,
                             fit_params=FIT_PARAMS,
                             eval_set_selection='test'
                             )
```
<img width="240" src="https://user-images.githubusercontent.com/59557625/146212543-8c49c900-eedb-453b-9c37-32eb3b25074a.png">

`eval_set_selection`引数については[こちら](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#-eval_data_sourceの指定値によるeval_setに入るデータの変化)を参照ください

### 4.3. チューニング実行
[3.で選択したチューニング用メソッド]()に対し、

・[1.で選択した評価指標](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#1-評価指標の選択)を`scoring`引数に

・[2.で選択したチューニング範囲](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#2-パラメータ探索範囲の選択)を`tuning_params`引数に

・[4.1で選択したクロスバリデーション手法](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#41-クロスバリデーション手法を選択)を`cv`引数に

指定し、実行します

（必要に応じて、4.2で選択した`fit_params`および`not_opt_params`引数も指定してください）

#### 実行例
Optunaでのチューニング実行例

```python
# 2.で選択したチューニング範囲を指定
TUNING_PARAMS = {'reg_alpha': (0.0001, 0.1),
                 'reg_lambda': (0.0001, 0.1),
                 'num_leaves': (2, 50),
                 'colsample_bytree': (0.4, 1.0),
                 'subsample': (0.4, 1.0),
                 'subsample_freq': (0, 7),
                 'min_child_samples': (0, 50)
                 }
# チューニング実行
best_params, best_score = tuning.optuna_tuning(scoring=SCORING,
                                               tuning_params=TUNING_PARAMS,
                                               cv=CV,
                                               not_opt_params=NOT_OPT_PARAMS,
                                               fit_params=FIT_PARAMS
                                               )
print(f'Best parameters\n{best_params}')  # 最適化されたパラメータ
print(f'Not tuned parameters\n{tuning.not_opt_params}')  # 最適化対象外パラメータ
print(f'Best score\n{best_score}')  # 最適パラメータでのスコア
print(f'Elapsed time\n{tuning.elapsed_time}')  # チューニング所要時間
```

実行結果

```
Best parameters
{'reg_alpha': 0.0016384726888678286, 'reg_lambda': 0.04644984234834465, 'num_leaves': 7, 'colsample_bytree': 0.7968135425414318, 'subsample': 0.7878217860051357, 'subsample_freq': 0, 'min_child_samples': 4}

Not tuned parameters
{'objective': 'regression', 'random_state': 42, 'boosting_type': 'gbdt', 'n_estimators': 10000}

Best score
-0.4124940780628491

Elapsed time
304.8196430206299
```

上記以外にも、チューニングの試行数や乱数シード、チューニング対象外のパラメータ等を引数として渡せます。
詳しくは以下のリンクを参照ください

|探索法|メソッドのAPI仕様リンク|
|---|---|
|グリッドサーチ|[grid_search_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#grid_search_tuningメソッド)|
|ランダムサーチ|[random_search_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)|
|ベイズ最適化 (BayesianOptimization)|[bayes_opt_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#bayes_opt_tuningメソッド)|
|ベイズ最適化 (Optuna)|[optuna_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#optuna_tuningメソッド)|

### 5.1. チューニング履歴の確認
[`plot_search_history()`]()メソッドでチューニング進行に伴うスコアの上昇履歴をグラフ表示し、スコアの上昇具合を確認します。

通常は進行とともに傾きが小さくなりますが、終了時点でもグラフの傾きが大きい場合、試行数を増やせばスコアが向上する可能性が高いです。

#### 実行例
Optunaでのチューニング実行後のチューニング履歴表示例

```python
tuning.plot_search_history()
```

実行結果

<img width="360" src="https://user-images.githubusercontent.com/59557625/146213822-f42bba74-bd6d-408a-a93b-d0c25f976253.png">

横軸は試行数以外に時間も指定できます(`x_axis`引数='time')

```python
tuning.plot_search_history(x_axis='time')
```

### 5.2. パラメータと評価指標の関係をマップ表示して確認
[`plot_search_history()`]()メソッドでパラメータと評価指標の関係をプロットし、評価指標のピークを捉えられているか確認します。
4.2で使用した手法がグリッドサーチならヒートマップで、それ以外なら散布図でプロットします。

パラメータが5次元以上のとき、以下の方法で表示軸を選択します。

・グリッドサーチ：パラメータの要素数()上位4パラメータを軸として表示します。表示軸以外のパラメータは最適値を使用します。

・グリッドサーチ以外：[`param_importances`]()の上位4パラメータを軸として表示します。

#### 実行例
Optunaでのチューニング実行後のパラメータと評価指標の関係表示

```python
tuning.plot_search_map()
```
実行結果

<img width="720" src="https://user-images.githubusercontent.com/59557625/146214163-d85658d6-3ff4-4df1-a564-48a7c77cb456.png">

### 5.3. 学習曲線を確認
[`plot_best_learning_curve()`]()メソッドで学習曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E5%AD%A6%E7%BF%92%E6%9B%B2%E7%B7%9A-1)「目的の性能を達成しているか」「過学習していないか」を確認します

#### 実行例
Optunaでのチューニング実行後の学習曲線を表示

```python
tuning.plot_best_learning_curve()
```
実行結果

<img width="360" src="https://user-images.githubusercontent.com/59557625/146214383-2c784424-173c-4c0c-b2c4-56e04bb1eb1b.png">

### 5.4. 検証曲線を確認
[`plot_best_validation_curve()`]()メソッドで検証曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E6%A4%9C%E8%A8%BC%E6%9B%B2%E7%B7%9A-2)「性能の最大値を捉えられているか」「過学習していないか」を確認します

#### 実行例
Optunaでのチューニング実行後の検証曲線を表示

```python
tuning.plot_best_validation_curve()
```
実行結果

<img width="800" src="https://user-images.githubusercontent.com/59557625/146214628-1fbddbe2-9198-4515-8536-2c1d858f7126.png">

### 6. チューニング後の学習器を使用する
チューニング後の学習器は`best_estimator`プロパティから取得できます。

また、学習器に`best_params`および`not_opt_params`プロパティの値を渡す事でも、チューニング後の学習器を再現することができます

後者の方法での実行例を下記します

#### 実行例
チューニング後の学習器から評価指標を求める (`best_score`プロパティと同じ値が求まります)

```python
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = LGBMRegressor(**params_after)
# 評価指標算出
scores = cross_val_score_eval_set('test',  # eval_set_selection
                            best_estimator, X, y,
                            scoring=tuning.scoring,
                            cv=tuning.cv,
                            fit_params=tuning.fit_params
                            )
print(np.mean(scores))
```
実行結果

```
-0.4124940780628491
```
<br>

チューニング前と同様、[seaborn-analyzer](https://pypi.org/project/seaborn-analyzer/)ライブラリを使用して予測値と実測値の関係を可視化すると、挙動がわかりやすくなるのでお勧めです

```python
from seaborn_analyzer import regplot
regplot.regression_pred_true(lgbmr,
                             x=tuning.x_colnames,
                             y='price',
                             data=df_boston,
                             scores='mse',
                             cv=tuning.cv,
                             fit_params=tuning.fit_params,
                             eval_set_selection=tuning.eval_set_selection
                             )
```

<img width="240" src="https://user-images.githubusercontent.com/59557625/146214802-22dfeb48-a9c8-4557-a6c6-d72581cd827d.png">

# MLflowによる結果ロギング
以下4種類のチューニング用メソッドの`mlflow_logging`引数を指定することで、MLflowで結果をロギングできます。

- [grid_search_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#grid_search_tuningメソッド)

- [random_search_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)

- [bayes_opt_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)

- [optuna_tuning()](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#random_search_tuningメソッド)

## ロギング時の引数指定
`mlflow_logging`引数の指定内容を変えることで、以下のようにロギング方法を変えることができます

|`mlflow_logging`引数|動作|
|---|---|
|None|MLflowによるロギングなし|
|'inside'|MLflow実行 (必要な処理を内部で全て自動実行)|
|'outside'|MLflow実行 (エクスペリメント立ち上げ等を手動実行する必要あり)|

## `mlflow_logging`='inside'のとき
`mlflow_logging`='inside'と指定すると、エクスペリメント立ち上げ等のMLflowに必要な処理が内部で自動実行されます。

引数指定以外の処理が不要で簡単にロギングできるため、通常はこちらの方法がおすすめです

### ・デフォルト構成での実行例
以下のように、チューニング用メソッド実行時に`mlflow_logging`='inside'を指定するとMLflowによるロギングが自動実行されます

```python
from tune_easy import LGBMRegressorTuning
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
# Optunaでのチューニング結果をMLflowでロギング
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用インスタンス作成
tuning.optuna_tuning(mlflow_logging='inside')  # MLflowのロギングを指定してOptunaでチューニング
```
チューニング完了後、ターミナルで以下のコマンドを打つと、MLflowのUI用Webサーバが立ち上がります。

```
mlflow ui
```
`mlflow_logging`[以外のMLflow用引数]()を指定していなければ、ローカルホストにUIが作成される（[こちらの記事]()のシナリオ1に相当）ので、ブラウザに`http://127.0.0.1:5000`と打つと、以下のような画面が表示されます。

<img width="800" src="https://user-images.githubusercontent.com/59557625/146404411-dc0102e9-e57c-49b4-82b4-0bc4a5c4bbf4.png">

各RUNの`Start Time`をクリックすると、保存されている情報の詳細を表示させることができます。詳細を後述します。

#### - RUNの保存内容
各RUNには、以下のような情報が保存されます

<img width="800" src="https://user-images.githubusercontent.com/59557625/147819096-fca5b4da-eca1-4898-bc04-592d74359571.png">

##### ・Parametersに保存される内容
以下の内容が記録されます

<img width="720" src="https://user-images.githubusercontent.com/59557625/147819346-1def6aa2-7f7a-4529-88b2-9abdc94014cb.png">

|名称|内容|値の取得元|再現方法|
|---|---|---|---|
|best__* |各パラメータの最適値|-|-|
|cv|クロスバリデーション分割法|[`cv`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`cv`引数|
|estimator|チューニング対象の学習器|[`estimator`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`estimator`引数|
|fit_params|学習時の`fit()`メソッドに渡す引数|[`fit_params`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`fit_params`引数|
|int_params|整数型のパラメータのリスト(ベイズ最適化系のみ)|[`int_params`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`int_params`引数|
|not_opt_params|チューニング対象外のパラメータ一覧|[`not_opt_params`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`not_opt_params`引数|
|param_scales|パラメータのスケール|[`param_scales`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`param_scales`引数|
|scoring|チューニングで最大化する評価スコア|[`scoring`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`scoring`引数|
|seed|乱数シード(クロスバリデーション分割、ベイズ最適化のサンプラー等で使用)|[`seed`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`seed`引数|
|tuning_params|チューニング対象のパラメータ一覧|[`tuning_params`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|チューニング用メソッド実行時の`tuning_params`引数|

##### ・Metricsに保存される内容
以下の内容が記録されます

<img width="200" src="https://user-images.githubusercontent.com/59557625/147820639-9e0d5be7-36de-494d-8dc0-ece3be7d33dd.png">

|名称|内容|
|---|---|
|elapsed_time|チューニングの所要時間（秒）|
|score_before|チューニング前のスコア（`scoring`で指定した指標）|
|score_best|チューニング後の最高スコア（`scoring`で指定した指標）|
|score_history|チューニング中のスコア推移（`scoring`で指定した指標）|

score_historyは、クリックすると推移をグラフ表示することができます

<img width="600" src="https://user-images.githubusercontent.com/59557625/147820829-2d40f343-cb99-4214-9c87-46e74fb680a3.png">

##### ・Tagsに保存される内容
以下の内容が記録されます

<img width="240" src="https://user-images.githubusercontent.com/59557625/147820868-e1a2f50c-1706-4064-9404-77cf705ab3cd.png">

|名称|内容|値の取得元|再現方法|
|---|---|---|---|
|tuning_algo|クロスバリデーション分割法|[`tuning_algo`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|実行するチューニング用メソッド（'optuna'と記録されていたら`optuna_tuning()`メソッドを実行）|
|x_colnames|説明変数の名称|[`x_colnames`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|クラス初期化時の`x_colnames`引数|
|y_colname|目的変数の名称|[`y_colname`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)の値|クラス初期化時の`y_colname`引数|

##### ・Artifactsに保存される内容
以下の内容が記録されます

<img width="640" alt="スクリーンショット 2021-12-31 19 32 54" src="https://user-images.githubusercontent.com/59557625/147821140-4e32bad9-434e-4c34-b157-d4654db03d79.png">

|名称|内容|
|---|---|
|best_estimator_* |最適パラメータを渡した学習器の[MLflow Modelsフォーマット](https://mlflow.org/docs/latest/models.html#storage-format)での保存データ|
|search_history.csv|探索履歴をCSV保存したファイル（[`search_history`プロパティ](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#プロパティ一覧)をDataFrame化して保存）|

### ・トラッキングサーバを指定した実行例
`mlflow_logging`[以外のMLflow用引数]()を指定すると、[トラッキングサーバ]()や[エクスペリメント名]()を指定したロギングが実行可能です

（トラッキングサーバやエクスペリメントについては[こちらの記事]()を参照ください）

リモートにトラッキングサーバを立ち上げた場合は、こちらの方法を利用ください

|引数名|指定内容|
|---|---|
|mlflow_tracking_uri|トラッキングサーバを指定|
|mlflow_artifact_location|Artifactストレージを指定※|
|mlflow_experiment_name|エクスペリメント名を指定|

※トラッキングサーバ作成時に[--default-artifact-root]()オプションを指定していれば、不要です。

[こちらの記事のシナリオ2の構成]()での実行例（SQLiteをバックエンドに指定）を下記します

```python
from tune_easy import LGBMRegressorTuning
import pandas as pd
import sqlite3
import os
# MLflow settings
DB_PATH = f'{os.getcwd()}/_tracking_uri/mlruns.db'  # バックエンドとなるSQLite DBファイルのパス
EXPERIMENT_NAME = 'optuna_regression'  # エクスペリメント名
ARTIFACT_LOCATION = f'{os.getcwd()}/_artifact_location'  # Artifactストレージ
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # トラッキングサーバ用フォルダ作成
conn = sqlite3.connect(DB_PATH)  # バックエンドとなるSQLite DBファイル作成
tracking_uri = f'sqlite:///{DB_PATH}'  # トラッキングサーバのURI
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIABLE = 'approval_rate'
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  
y = df_reg[TARGET_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
# Optunaでのチューニング結果をMLflowでロギング
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用インスタンス作成
tuning.optuna_tuning(mlflow_logging='inside')  # MLflowのロギングを指定してOptunaでチューニング
```

## `mlflow_logging`='outside'のとき
`mlflow_logging`='outside'と指定すると、エクスペリメント立ち上げ等のMLflowに必要な処理を外部で明示的に実装する必要があります。

本ライブラリで自動記録されるもの以外の情報も同じRunで記録したい場合、こちらの方法を使用してください

### ・実装例

```python
from tune_easy import SVMRegressorTuning
import pandas as pd
import mlflow
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
TARGET_VARIALBLE_REG = 'approval_rate'  # 目的変数
USE_EXPLANATORY_REG = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[TARGET_VARIALBLE_REG].values
X = df_reg[USE_EXPLANATORY_REG].values
tuning = SVMRegressorTuning(X, y, USE_EXPLANATORY_REG, y_colname=TARGET_VARIALBLE_REG)
# MLflowのRun開始
with mlflow.start_run() as run:
    # Optunaでのチューニング結果をMLflowでロギング
    tuning.optuna_tuning(mlflow_logging='outside')
    # 追加で記録したい情報
    mlflow.log_param('data_name', 'osaka_metropolis')
    mlflow.log_dict(tuning.tuning_params, 'tuning_params.json')
```
※ なお、この方法では`mlflow_logging`以外のMLflow用引数 (`mlflow_tracking_uri`, `mlflow_artifact_location`, `mlflow_experiment_name`)は指定できないのでご注意ください。
