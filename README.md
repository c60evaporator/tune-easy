# param-tuning-utility
**A hyperparameter tuning tool with gorgeous UI for scikit-learn API.**

This documentation is Japanese language version.
**[English version is here]()**

**[API reference is here]()**

<br>

# 使用法
LightGBM回帰のOptunaによるチューニング実行例

```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target  # 目的変数
###### チューニング実行と評価 ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス
tuning.plot_first_validation_curve()  # 範囲を定めて検証曲線をプロット
tuning.optuna_tuning()  # Optunaによるチューニング実行
tuning.plot_search_history()  # Optuna実行
tuning.plot_search_map()  # 探索点と評価指標を可視化
tuning.plot_best_learning_curve()  # 学習曲線の可視化
tuning.plot_best_validation_curve()  # 検証曲線の可視化
```
上例は引数を指定せずにデフォルト設定でチューニングを実施しています。

デフォルト設定でも多くの場合は適切にチューニングできますが、設定を調整しながら丁寧に実行したい場合は、[チューニング手順](https://github.com/c60evaporator/param-tuning-utility#%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E6%89%8B%E9%A0%86)の項を参照ください。

<br>

# 必要要件
* Python >=3.6
* Scikit-learn >=0.24.2
* Numpy >=1.20.3
* Pandas >=1.2.4
* Matplotlib >=3.3.4
* Seaborn >=0.11.0
* Optuna >=2.7.0
* BayesianOptimization >=1.2.0
* MLFlow >=1.17.0
* LightGBM >=3.2.1
* XGBoost >=1.4.2
<br>

# インストール方法
```
$ pip install param-tuning-utility
```
<br>

# サポート
バグ等は[Issues](https://github.com/c60evaporator/param-tuning-utility/issues)で報告してください

<br>

# チューニング手順
**下図の手順**([こちらの記事に詳細](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#2-%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%AE%E6%89%8B%E9%A0%86%E3%81%A8%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E4%B8%80%E8%A6%A7))に従い、パラメータチューニングを実施します。

Scikit-LearnのAPIに対応した学習器が対象となります。

![image](https://user-images.githubusercontent.com/59557625/130362754-a85fc7fc-38f7-4d5a-9e8f-c24321a2ed98.png)

## 手順一覧
**0. チューニングの準備**

&nbsp;├─ [0.1. データの読込＆前処理](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#01-%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E8%BE%BC%E5%89%8D%E5%87%A6%E7%90%86)

&nbsp;└─ [0.2. チューニング用クラスの初期化](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#02-%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E7%94%A8%E3%82%AF%E3%83%A9%E3%82%B9%E3%81%AE%E5%88%9D%E6%9C%9F%E5%8C%96)

**1. 最大化したい評価指標を定義**

&nbsp;└─ [1.1. 評価指標の選択](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#1-%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99%E3%81%AE%E9%81%B8%E6%8A%9E)

**2. パラメータ探索範囲の選択**

&nbsp;└─ [2.1. パラメータ探索範囲の選択](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#2-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E6%8E%A2%E7%B4%A2%E7%AF%84%E5%9B%B2%E3%81%AE%E9%81%B8%E6%8A%9E)

**3. 探索法を選択**

&nbsp;└─ [3.1. 探索手法を選択](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#3-%E6%8E%A2%E7%B4%A2%E6%B3%95%E3%82%92%E9%81%B8%E6%8A%9E)

**4. クロスバリデーションでチューニングを実行**

&nbsp;├─ [4.1. クロスバリデーション手法を選択](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#41-%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E6%89%8B%E6%B3%95%E3%82%92%E9%81%B8%E6%8A%9E)

&nbsp;├─ [4.2. チューニング前のスコアを確認](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#42-%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E5%89%8D%E3%81%AE%E3%82%B9%E3%82%B3%E3%82%A2%E3%82%92%E7%A2%BA%E8%AA%8D)

&nbsp;└─ [4.3. チューニング実行](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#43-%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E5%AE%9F%E8%A1%8C)

**5. 学習曲線・検証曲線等でチューニング結果を確認**

&nbsp;├─ [5.1. チューニング履歴の確認](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#01-%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E8%BE%BC%E5%89%8D%E5%87%A6%E7%90%86)

&nbsp;├─ [5.2. パラメータと評価指標の関係を確認](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#52-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%A8%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99%E3%81%AE%E9%96%A2%E4%BF%82%E3%82%92%E7%A2%BA%E8%AA%8D)

&nbsp;├─ [5.3. 学習曲線を確認](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#53-%E5%AD%A6%E7%BF%92%E6%9B%B2%E7%B7%9A%E3%82%92%E7%A2%BA%E8%AA%8D)

&nbsp;└─ [5.4. 検証曲線を確認](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#54-%E6%A4%9C%E8%A8%BC%E6%9B%B2%E7%B7%9A%E3%82%92%E7%A2%BA%E8%AA%8D)

**6. チューニング結果の活用**

&nbsp;└─ [6.1. チューニング後の学習器を使用する](https://github.com/c60evaporator/param-tuning-utility/blob/master/README.md#6-%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E5%BE%8C%E3%81%AE%E5%AD%A6%E7%BF%92%E5%99%A8%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%99%E3%82%8B)

<br>

## 0.1. データの読込＆前処理
使用するデータを読み込み、特徴量選択等の前処理を実行します。

### 実行例
ボストン住宅価格データセットを読み込み、特徴量 (説明変数)を選択します
```python
from sklearn.datasets import load_boston
import pandas as pd

USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 選択した5つの説明変数
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target  # 目的変数
```

※選択した5特徴量は、ランダムフォレストのRFE (再帰的特徴量削減)により選定
```python
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

X_all = load_boston().data
y = load_boston().target
selector = RFE(RandomForestRegressor(random_state=42), n_features_to_select=5)
selector.fit(X_all, y)
print(load_boston().feature_names)
print(selector.get_support())
```

```
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
[ True False False False  True  True False  True False False False False
  True]
```
特徴量選択については、[Scikit-Learn公式](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)を参照ください

## 0.2. チューニング用クラスの初期化
以下を参考に使用したい学習器に合わせてチューニング用クラスを選択し、インスタンスを作成します

|学習器の種類|クラス名|
|---|---|
|LightGBM回帰||LGBMRegressorTuning|
|XGBRegressorTuning|XGBRegressorTuning|
|サポートベクター回帰|SVMRegressorTuning|
|ランダムフォレスト回帰|RFRegressorTuning|
|ElasticNet回帰|ElasticNetTuning|
|LightGBM分類|LGBMClassifierTuning|
|XGBoost分類|XGBClassifierTuning|
|サポートベクターマシン分類|SVMClassifierTuning|
|ランダムフォレスト分類|RFClassifierTuning|

### 実行例
LightGBM回帰のチューニング用クラス初期化
```python
from param_tuning import LGBMRegressorTuning
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)
```

## 1. 評価指標の選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#21-%E8%A9%95%E4%BE%A1%E6%8C%87%E6%A8%99%E3%81%AE%E5%AE%9A%E7%BE%A9)チューニングの評価指標を選択します。

デフォルト(各メソッドの`scoring`引数を指定しないとき)では、以下の指標を使用します
|手法|デフォルトで使用する手法|
|---|---|
|回帰|RMSE ('neg_mean_squared_error')|
|2クラス分類|LogLoss ('neg_log_loss')|
|多クラス分類|LogLoss ('neg_log_loss')|

### 実行例
RMSEを指標に使用するとき
```
SCORING = 'neg_mean_squared_error'
```

## 2. パラメータ探索範囲の選択
[`plot_first_validation_curve()`]()メソッドで検証曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#22-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E7%A8%AE%E9%A1%9E%E3%81%A8%E6%8E%A2%E7%B4%A2%E7%AF%84%E5%9B%B2%E3%81%AE%E9%81%B8%E6%8A%9E)パラメータ探索範囲を選択します

事前に[4.1. クロスバリデーション手法の選択]()を実施し、`cv`引数に指定する事が望ましいです

### 実行例
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

![image](https://user-images.githubusercontent.com/59557625/130490027-5ff1b717-7e45-4e02-8e50-79fd6e49b19f.png)

下図のように検証曲線から過学習にも未学習にもなりすぎていない範囲を抽出し、探索範囲とすることが望ましいです (数はSVMでの実行例)

![image](https://user-images.githubusercontent.com/59557625/130347923-c3ed17a2-8ad6-4f30-8ff7-fc91cf8f97ee.png)

## 3. 探索法を選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E6%89%8B%E9%A0%8634-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E9%81%B8%E6%8A%9E%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)パラメータの探索法を選択します。
以下の4種類の探索法から使用したい手法を選び、対応したメソッドを選択します

|探索法|使用するメソッド|
|---|---|
|グリッドサーチ|[grid_search_tuning()]()|
|ランダムサーチ|[random_search_tuning()]()|
|ベイズ最適化 (BayesianOptimization)|[bayes_opt_tuning()]()|
|ベイズ最適化 (Optuna)|[optuna_tuning()]()|

## 4.1. クロスバリデーション手法を選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#24-%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)クロスバリデーションの手法を選択します。

### 実行例
クロスバリデーションに5分割KFoldを指定
```python
CV = KFold(n_splits=5, shuffle=True, random_state=42)
```

## 4.2. チューニング前のスコアを確認
`※チューニング前スコアは4.3実行時にも表示されるので、本工程は飛ばしても構いません`

チューニング前のスコアを確認します。
このとき

・学習器の`fit()`メソッドに渡す引数（LightGBMの`early_stopping_rounds`等）

・チューニング対象外パラメータを学習器インスタンスに渡す事

を忘れないようにご注意ください

### 実行例
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
-11.979161807916636
```
<br>

[seaborn-analyzer](https://pypi.org/project/seaborn-analyzer/)ライブラリを活用して予測値と実測値の関係を可視化すると、挙動がわかりやすくなるのでお勧めです
```python
from seaborn_analyzer import regplot
df_boston['price'] = y
regplot.regression_pred_true(lgbmr,
                             x=tuning.x_colnames,
                             y='price',
                             data=df_boston,
                             scores='mse',
                             cv=CV,
                             fit_params=FIT_PARAMS
                             )
```
![image](https://user-images.githubusercontent.com/59557625/130487845-2c9db099-f137-489a-9b09-9f01a8d55f1e.png)

## 4.3. チューニング実行
[3.で選択したチューニング用メソッド]()に対し、

・[1.で選択した評価指標]()を`scoring`引数に

・[2.で選択したチューニング範囲]()を`tuning_params`引数に

・[4.1で選択したクロスバリデーション手法]()を`cv`引数に

指定し、実行します

（必要に応じて、4.2で選択した`fit_params`および`not_opt_params`引数も指定してください）

### 実行例
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
{'reg_alpha': 0.003109527801280432, 'reg_lambda': 0.0035808676982557147, 'num_leaves': 41, 'colsample_bytree': 0.9453510369496361, 'subsample': 0.5947574986660598, 'subsample_freq': 1, 'min_child_samples': 0}

Not tuned parameters
{'objective': 'regression', 'random_state': 42, 'boosting_type': 'gbdt', 'n_estimators': 10000}

Best score
-9.616609903204923

Elapsed time
278.3654930591583
```

上記以外にも、チューニングの試行数や乱数シード、チューニング対象外のパラメータ等を引数として渡せます。
詳しくは以下のリンクを参照ください

|探索法|メソッドの引数リンク|
|---|---|
|グリッドサーチ|[grid_search_tuning()]()|
|ランダムサーチ|[random_search_tuning()]()|
|ベイズ最適化 (BayesianOptimization)|[bayes_opt_tuning()]()|
|ベイズ最適化 (Optuna)|[optuna_tuning()]()|

## 5.1. チューニング履歴の確認
[`plot_search_history()`]()メソッドでチューニング進行に伴うスコアの上昇履歴をグラフ表示し、スコアの上昇具合を確認します。

通常は進行とともに傾きが小さくなりますが、終了時点でもグラフの傾きが大きい場合、試行数を増やせばスコアが向上する可能性が高いです。

### 実行例
Optunaでのチューニング実行後のチューニング履歴表示例
```python
tuning.plot_search_history()
```

実行結果

![image](https://user-images.githubusercontent.com/59557625/130488044-75d316ba-f251-4ecd-9729-65d33e402b5b.png)

横軸は試行数以外に時間も指定できます(`x_axis`引数='time')

```python
tuning.plot_search_history(x_axis='time')
```

## 5.2. パラメータと評価指標の関係を確認
[`plot_search_history()`]()メソッドでパラメータと評価指標の関係をプロットし、評価指標のピークを捉えられているか確認します。
4.2で使用した手法がグリッドサーチならヒートマップで、それ以外なら散布図でプロットします。

パラメータが5次元以上のとき、以下の方法で表示軸を選択します。

・グリッドサーチ：パラメータの要素数()上位4パラメータを軸として表示します。表示軸以外のパラメータは最適値を使用します。

・グリッドサーチ以外：[後述の`param_importances`]()の上位4パラメータを軸として表示します。

### 実行例
Optunaでのチューニング実行後のパラメータと評価指標の関係表示
```python
tuning.plot_search_map()
```
実行結果

![image](https://user-images.githubusercontent.com/59557625/130488301-da358b25-5ba9-4306-8e76-5e28153b89d2.png)

## 5.3. 学習曲線を確認
[`plot_best_learning_curve()`]()メソッドで学習曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E5%AD%A6%E7%BF%92%E6%9B%B2%E7%B7%9A-1)「目的の性能を達成しているか」「過学習していないか」を確認します

### 実行例
Optunaでのチューニング実行後の学習曲線を表示
```python
tuning.plot_best_learning_curve()
```
実行結果

![image](https://user-images.githubusercontent.com/59557625/130488483-387b6fdb-8ca3-47d6-ad25-d142463683b0.png)

## 5.4. 検証曲線を確認
[`plot_best_validation_curve()`]()メソッドで検証曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E6%A4%9C%E8%A8%BC%E6%9B%B2%E7%B7%9A-2)「性能の最大値を捉えられているか」「過学習していないか」を確認します

### 実行例
Optunaでのチューニング実行後の検証曲線を表示
```python
tuning.plot_best_validation_curve()
```
実行結果

![image](https://user-images.githubusercontent.com/59557625/130490273-345dbc31-2201-4752-be79-0749058c2b00.png)

## 6. チューニング後の学習器を使用する
チューニング後の学習器は`best_estimator`プロパティから取得できます。

また、学習器に`best_params`および`not_opt_params`プロパティの値を渡す事でも、チューニング後の学習器を再現することができます

後者の方法での実行例を下記します

### 実行例
チューニング後の学習器から評価指標を求める (`best_score`プロパティと同じ値が求まります)
```python
params_after = {}
params_after.update(tuning.best_params)
params_after.update(tuning.not_opt_params)
best_estimator = LGBMRegressor(**params_after)
# 評価指標算出
scores = cross_val_score(best_estimator, X, y,
                         scoring=tuning.scoring,
                         cv=tuning.cv,
                         fit_params=tuning.fit_params
                         )
print(np.mean(scores))
```
実行結果
```
-9.616609903204923
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
                             fit_params=tuning.fit_params
                             )
```
![image](https://user-images.githubusercontent.com/59557625/130488697-ab6f83f9-3895-4ec6-a761-770f5396bd0e.png)

<br>
<br>

# クラス一覧
以下のクラスからなります
|クラス名|パッケージ名|概要|使用法|
|---|---|---|---|
|LGBMRegressorTuning|lgbm_tuning.py|LightGBM回帰のパラメータチューニング用クラス|[リンク]()|
|XGBRegressorTuning|xgb_tuning.py|XGBoost回帰のパラメータチューニング用クラス|[リンク]()|
|SVMRegressorTuning|svm_tuning.py|サポートベクター回帰のパラメータチューニング用クラス|[リンク]()|
|RFRegressorTuning|rf_tuning.py|ランダムフォレスト回帰のパラメータチューニング用クラス|[リンク]()|
|ElasticNetTuning|elasticnet_tuning.py|ElasticNet回帰のパラメータチューニング用クラス|[リンク]()|
|LGBMClassifierTuning|lgbm_tuning.py|LightGBM分類のパラメータチューニング用クラス|[リンク]()|
|XGBClassifierTuning|xgb_tuning.py|XGBoost分類のパラメータチューニング用クラス|[リンク]()|
|SVMClassifierTuning|svm_tuning.py|サポートベクターマシン分類のパラメータチューニング用クラス|[リンク]()|
|RFClassifierTuning|rf_tuning.py|ランダムフォレスト分類のパラメータチューニング用クラス|[リンク]()|

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
|eval_data_source|オプション|{'all', 'valid', 'train'}|'all'|XGBoost, LightGBMにおける`fit_params`の`eval_set`の指定方法, 'all'ならeval_set =[(self.X, self.y)]|

### 実行例
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしで初期化
LightGBM回帰におけるクラス初期化実行例
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = boston[USE_EXPLANATORY].values
y = load_boston().target
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
OBJECTIVE_VARIABLE = 'approval_rate'  # 目的変数
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLATATORY].values
###### クラス初期化 ######
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY,  # 必須引数
                             cv_group=df_reg['ward_after'].values)  # グルーピング対象データ (大阪都構想の区)
```

#### 検証データをfit_paramsのeval_setに使用したいとき
デフォルトではeval_set (early_stopping_roundの判定に使用するデータ)は全てのデータ (self.X, self.y)を使用しますが、eval_data_source='valid'を指定するとクロスバリデーションの検証用データのみを使用します
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = boston[USE_EXPLANATORY].values
y = load_boston().target
###### クラス初期化 ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY,
                             eval_data_source='valid')  # eval_setの指定方法 (検証用データを渡す)
```

<br>

# メソッド一覧
上記クラスは、以下のようなメソッドを持ちます

|メソッド名|機能|
|---|---|
|plot_first_validation_curve|範囲を定めて検証曲線をプロットし、パラメータ調整範囲の参考とします|
|grid_search_tuning|グリッドサーチを実行します|
|random_search_tuning|ランダムサーチを実行します|
|bayes_opt_tuning|BayesianOptimizationでベイズ最適化を実行します|
|optuna_tuning|Optunaでベイズ最適化を実行します|
|plot_search_history|チューニング進行に伴うスコアの上昇履歴をグラフ表示します|
|get_search_history|チューニング進行に伴うスコアの上昇履歴をpandas.DataFrameで取得します|
|plot_search_map|チューニング探索点と評価指標をマップ表示します (グリッドサーチはヒートマップ、それ以外は散布図)|
|plot_best_learning_curve|学習曲線をプロットし、チューニング後モデルのバイアスとバリアンスの判断材料とします|
|plot_best_validation_curve|学習後の検証曲線をプロットし、チューニング妥当性の判断材料とします|
|plot_param_importances|パラメータを説明変数としてスコアをランダムフォレスト回帰した際のfeature_importancesを取得します。スコアの変化に寄与するパラメータの判断材料とします|
|get_feature_importances|チューニング後の最適モデルのfeature_importancesを取得します (feature_importances対応学習器のみ)|
|plot_feature_importances|チューニング後の最適モデルのfeature_importancesをグラフ表示します (feature_importances対応学習器のみ)|

※大半のメソッドは全てのクラスに対応していますが、
get_feature_importancesおよびplot_feature_importancesメソッドは、XGBoostおよびLightGBMのみ対応しています。

## plot_first_validation_curveメソッド
範囲を定めて検証曲線をプロットし、パラメータ調整範囲の参考とします

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なる]()|最適化対象の学習器インスタンス。`not_opt_parans`で指定したパラメータは上書きされるので注意|
|validation_<br>curve_params|オプション　　|dict[str, list[float]]|[クラスごとに異なる]()|検証曲線プロット対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|not_opt_params|オプション|dict|[クラスごとに異なる]()|`validation_curve_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なる]()|`validation_curve_params`のパラメータごとのスケール('linear', 'log')|
|plot_stats|オプション|str|'mean'|検証曲線グラフにプロットする統計値 ('mean'(平均±標準偏差), 'median'(中央値&最大最小値))|
|axes|オプション|list[matplotlib.axes.Axes]|None|グラフ描画に使用するaxes (Noneならmatplotlib.pyplot.plotで1枚ごとにプロット)|
|fit_params|オプション|dict|[クラスごとに異なる]()|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしで検証曲線プロット
オプション引数を指定しないとき、[前述のデフォルト値]()を使用してプロットします
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数で検証曲線プロット ######
tuning.plot_first_validation_curve()
```
実行結果

![image](https://user-images.githubusercontent.com/59557625/130490027-5ff1b717-7e45-4e02-8e50-79fd6e49b19f.png)

#### パラメータ範囲を指定して検証曲線プロット
`validation_curve_params`引数で、検証曲線のパラメータ範囲を指定する事ができます
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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

![image](https://user-images.githubusercontent.com/59557625/130651966-5c78f390-6bb0-474e-b64b-1b3c34eb0943.png)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L123)をご参照ください

<br>
<br>

## grid_search_tuningメソッド
グリッドサーチを実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なる]()|最適化対象の学習器インスタンス。`not_opt_parans`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なる]()|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error' in regression.'neg_log_loss' in clasification|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なる]()|`tuning_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なる]()|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_logging|オプション|str|None|MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)。詳細は[こちら]()|
|grid_kws|オプション|dict|None|sklearn.model_selection.GridSearchCVに渡す引数 (estimator, tuning_params, cv, scoring以外)|
|fit_params|オプション|dict|[クラスごとに異なる]()|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしでグリッドサーチ
オプション引数を指定しないとき、[デフォルトの引数]()を使用してプロットします
```python
from param_tuning import RFRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でグリッドサーチ ######
best_params, best_score = tuning.grid_search_tuning()
```
実行結果
```
score before tuning = -11.719820569093374
best_params = {'max_depth': 32, 'max_features': 2, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 160}
score after tuning = -10.497362132823111
```

#### パラメータ探索範囲を指定してグリッドサーチ
`tuning_params`引数で、グリッドサーチのパラメータ探索範囲を指定する事ができます
```python
from param_tuning import RFRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
score before tuning = -11.719820569093374
best_params = {'max_depth': 32, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 80}
score after tuning = -11.621063650183345
```

#### 学習器を指定してグリッドサーチ
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です
```python
from param_tuning import RFRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
score before tuning = -11.724246256998635
best_params = {'rf__max_depth': 32, 'rf__max_features': 2, 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 160}
score after tuning = -10.477565908068511
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
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なる]()|最適化対象の学習器インスタンス。`not_opt_parans`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なる]()|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|n_iter|オプション|int|[クラスごとに異なる]()|ランダムサーチの試行数|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なる]()|`tuning_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なる]()|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_<br>logging|オプション|str|None|MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)。詳細は[こちら]()|
|rand_kws|オプション|dict|None|sklearn.model_selection.RondomizedSearchCVに渡す引数 (estimator, tuning_params, cv, scoring, n_iter以外)|
|fit_params|オプション|dict|[クラスごとに異なる]()|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしでランダムサーチ
オプション引数を指定しないとき、[デフォルトの引数]()を使用してプロットします
```python
from param_tuning import RFRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = RFRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でランダムサーチ ######
best_params, best_score = tuning.random_search_tuning()
```
実行結果
```
score before tuning = -11.719820569093374
best_params = {'n_estimators': 160, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 2, 'max_depth': 8}
score after tuning = -10.832494601564617
```

#### パラメータ探索範囲と試行数を指定してランダムサーチ
`tuning_params`引数で、ランダムサーチのパラメータ探索範囲を指定する事ができます。

また、`n_iter`引数で探索の試行数を指定できます
```python
from param_tuning import RFRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
score before tuning = -11.719820569093374
best_params = {'n_estimators': 30, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 2, 'max_depth': 32}
score after tuning = -10.890781333521025
```

#### 学習器を指定してランダムサーチ
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です
```python
from param_tuning import RFRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
score before tuning = -11.724246256998635
best_params = {'rf__n_estimators': 30, 'rf__min_samples_split': 8, 'rf__min_samples_leaf': 1, 'rf__max_features': 2, 'rf__max_depth': 32}
score after tuning = -10.84662079640907
```
※本来パイプラインのパラメータ名は`学習器名__パラメータ名`と指定する必要がありますが、本ツールの`tuning_params`には自動で学習器名を付加する機能を追加しているので、`パラメータ名`のみでも指定可能です (`fit_params`指定時も同様)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_svr_regression.py#L155)をご参照ください

<br>
<br>

## bayes_opt_tuningメソッド
[BayesianOptimizationライブラリ]()によるベイズ最適化を実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なる]()|最適化対象の学習器インスタンス。`not_opt_parans`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なる]()|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (BayesianOptimization初期化時の`random_state`引数、および学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|n_iter|オプション|int|[クラスごとに異なる]()|ベイズ最適化の試行数|
|init_points|オプション|int|[クラスごとに異なる]()|ランダムな初期探索点の個数|
|acq|オプション|{'ei', 'pi', 'ucb'}|'ei'|獲得関数 ('ei': EI戦略, 'pi': PI戦略, 'ucb': UCB戦略)|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なる]()|`tuning_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なる]()|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_<br>logging|オプション|str|None|MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)。詳細は[こちら]()|
|fit_params|オプション|dict|[クラスごとに異なる]()|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしでBayesianOptimization
オプション引数を指定しないとき、[デフォルトの引数]()を使用してBayesianOptimizationでチューニングします
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でBayesianOptimization ######
best_params, best_score = tuning.bayes_opt_tuning()
```
実行結果
```
score before tuning = -11.979161807916636

|   iter    |  target   | colsam... | min_ch... | num_le... | reg_alpha | reg_la... | subsample | subsam... |
-------------------------------------------------------------------------------------------------------------
|  1        | -14.19    |  0.6247   |  47.54    |  37.14    | -2.204    | -3.532    |  0.4936   |  0.4066   |
|  2        | -12.78    |  0.9197   |  30.06    |  35.99    | -3.938    | -1.09     |  0.8995   |  1.486    |
  :
  :
|  70       | -13.24    |  0.9837   |  33.73    |  20.4     | -3.098    | -2.831    |  0.4122   |  0.0941   |
=============================================================================================================
best_params = {'colsample_bytree': 0.9479943197172334, 'min_child_samples': 14, 'num_leaves': 8, 'reg_alpha': 0.011998136674904547, 'reg_lambda': 0.03204706412377387, 'subsample': 0.595559561303092, 'subsample_freq': 6}
score after tuning = -11.488951692728644
```

#### パラメータ範囲と試行数を指定してBayesianOptimization
`tuning_params`引数で、ベイズ最適化のパラメータ探索範囲を指定する事ができます。

また、`n_iter`引数でベイズ最適化の試行数を指定できます

探索の合計試行数は、`init_points`で指定したランダム初期点数 + `n_iter`となります
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
|  1        | -12.14    |  0.6247   |  19.01    |  37.14    | -1.803    | -2.688    |  0.4624   |  0.2904   |
|  2        | -11.58    |  0.9197   |  12.02    |  35.99    | -2.959    | -1.06     |  0.733    |  1.062    |
  :
  :
|  85       | -10.5     |  1.0      |  2.584    |  16.71    | -1.364    | -1.615    |  0.8      |  3.498    |
=============================================================================================================
best_params = {'colsample_bytree': 1.0, 'min_child_samples': 3, 'num_leaves': 17, 'reg_alpha': 0.04321535571671176, 'reg_lambda': 0.024267598888301777, 'subsample': 0.8, 'subsample_freq': 3}
score after tuning = -10.497077015105946
```

#### 学習器を指定してBayesianOptimization
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
score before tuning = -12.255823372962741
|   iter    |  target   | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... | lgbmr_... |
-------------------------------------------------------------------------------------------------------------
|  1        | -11.99    |  0.6247   |  19.01    |  37.14    | -1.803    | -2.688    |  0.4624   |  0.2904   |
|  2        | -11.53    |  0.9197   |  12.02    |  35.99    | -2.959    | -1.06     |  0.733    |  1.062    |
  :
  :
|  85       | -12.36    |  0.8191   |  0.9298   |  31.81    | -2.811    | -1.314    |  0.4321   |  4.011    |
=============================================================================================================
best_params = {'lgbmr__colsample_bytree': 0.9248812273481077, 'lgbmr__min_child_samples': 4, 'lgbmr__num_leaves': 8, 'lgbmr__reg_alpha': 0.00329559380608668, 'lgbmr__reg_lambda': 0.012905694418620333, 'lgbmr__subsample': 0.5099088770971258, 'lgbmr__subsample_freq': 1}
score after tuning = -10.937025098477642
```
※本来パイプラインのパラメータ名は`学習器名__パラメータ名`と指定する必要がありますが、本ツールの`tuning_params`には自動で学習器名を付加する機能を追加しているので、`パラメータ名`のみでも指定可能です (`fit_params`指定時も同様)

<br>

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_svr_regression.py#L202)をご参照ください

<br>
<br>

## optuna_tuningメソッド
[Optunaライブラリ]()によるベイズ最適化を実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|estimator|オプション|estimator object implementing 'fit'|[クラスごとに異なる]()|最適化対象の学習器インスタンス。`not_opt_parans`で指定したパラメータは上書きされるので注意|
|tuning_params|オプション|dict[str, list[float]]|[クラスごとに異なる]()|チューニング対象のパラメータ範囲|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|seed|オプション|int|42|乱数シード (TPESamplerの`seed`引数、および学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|scoring|オプション|str|'neg_mean_squared_error'|最適化で最大化する評価指標 ('neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_log_loss', 'f1'など)|
|n_trials|オプション|int|[クラスごとに異なる]()|ベイズ最適化の試行数|
|study_kws|オプション|dict|{'sampler': TPESampler(), 'direction': 'maximize'}|optuna.study.create_study()に渡す引数|
|optimize_kws|オプション|dict|{}|optuna.study.Study.optimize()に渡す引数 (n_trials以外)|
|not_opt_<br>params|オプション　　|dict|[クラスごとに異なる]()|`tuning_params`以外のチューニング対象外パラメータを指定|
|param_scales|オプション|dict[str, str]|[クラスごとに異なる]()|`tuning_params`のパラメータごとのスケール('linear', 'log')|
|mlflow_<br>logging|オプション|str|None|MLFlowでの結果記録有無('log':通常の記録, 'with':with構文で記録, None:記録なし)。詳細は[こちら]()|
|fit_params|オプション|dict|[クラスごとに異なる]()|学習器の`fit()`メソッドに渡すパラメータ|

### 実行例
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしでOptunaチューニング
オプション引数を指定しないとき、[デフォルトの引数]()を使用してOptunaでチューニングします
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### デフォルト引数でOptunaチューニング ######
best_params, best_score = tuning.optuna_tuning()
```
実行結果
```
[I 2021-08-27 01:02:55,935] A new study created in memory with name: no-name-bdd17556-7ee0-48bd-a219-c331b5f83de3
score before tuning = -11.979161807916636

[I 2021-08-27 01:02:57,904] Trial 0 finished with value: -11.452939275675764 and parameters: {'reg_alpha': 0.0013292918943162175, 'reg_lambda': 0.07114476009343425, 'num_leaves': 37, 'colsample_bytree': 0.759195090518222, 'subsample': 0.4936111842654619, 'subsample_freq': 1, 'min_child_samples': 2}. Best is trial 0 with value: -11.452939275675764.
  :
  :
[I 2021-08-27 00:57:48,371] Trial 199 finished with value: -10.511358593135807 and parameters: {'reg_alpha': 0.0010789348786651755, 'reg_lambda': 0.00045050344430450247, 'num_leaves': 42, 'colsample_bytree': 0.9456531771486588, 'subsample': 0.6186892820565172, 'subsample_freq': 1, 'min_child_samples': 1}. Best is trial 188 with value: -9.616609903204923.
best_params = {'reg_alpha': 0.003109527801280432, 'reg_lambda': 0.0035808676982557147, 'num_leaves': 41, 'colsample_bytree': 0.9453510369496361, 'subsample': 0.5947574986660598, 'subsample_freq': 1, 'min_child_samples': 0}
score after tuning = -9.616609903204923
```

#### パラメータ範囲と試行数を指定してOptunaチューニング実行
`tuning_params`引数で、ベイズ最適化のパラメータ探索範囲を指定する事ができます。

また、`n_trials`引数でベイズ最適化の試行数を指定できます

```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
[I 2021-08-27 01:15:21,268] A new study created in memory with name: no-name-635ede1b-7985-4548-aba8-c831963b9a8c
score before tuning = -11.979161807916636

[I 2021-08-27 01:15:23,498] Trial 0 finished with value: -12.110663238032412 and parameters: {'reg_alpha': 0.005611516415334507, 'reg_lambda': 0.07969454818643935, 'num_leaves': 37, 'colsample_bytree': 0.759195090518222, 'subsample': 0.46240745617697465, 'subsample_freq': 0, 'min_child_samples': 1}. Best is trial 0 with value: -12.110663238032412.
  :
  :
[I 2021-08-27 01:12:09,708] Trial 199 finished with value: -10.194230069453065 and parameters: {'reg_alpha': 0.001828306094799145, 'reg_lambda': 0.06571376528457373, 'num_leaves': 11, 'colsample_bytree': 0.9278346121998875, 'subsample': 0.7163240012735217, 'subsample_freq': 1, 'min_child_samples': 1}. Best is trial 173 with value: -9.823907953731936.
best_params = {'reg_alpha': 0.0027344867053618453, 'reg_lambda': 0.06607544948281772, 'num_leaves': 10, 'colsample_bytree': 0.9598501478819825, 'subsample': 0.6910946770860599, 'subsample_freq': 1, 'min_child_samples': 1}
score after tuning = -9.823907953731936
```

#### 学習器を指定してOptunaチューニング実行
`estimator`引数で、学習器を指定する事ができます。パイプラインも指定可能です
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
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
[I 2021-08-27 01:24:57,636] A new study created in memory with name: no-name-2fb193f2-351c-4d4c-baae-b23d16859665
score before tuning = -12.255823372962741

[I 2021-08-27 01:25:01,806] Trial 0 finished with value: -11.378859002518542 and parameters: {'lgbmr__reg_alpha': 0.005611516415334507, 'lgbmr__reg_lambda': 0.07969454818643935, 'lgbmr__num_leaves': 37, 'lgbmr__colsample_bytree': 0.759195090518222, 'lgbmr__subsample': 0.46240745617697465, 'lgbmr__subsample_freq': 0, 'lgbmr__min_child_samples': 1}. Best is trial 0 with value: -11.378859002518542.
  :
  :
[I 2021-08-27 01:22:03,721] Trial 199 finished with value: -10.096037009116241 and parameters: {'lgbmr__reg_alpha': 0.002327721686775674, 'lgbmr__reg_lambda': 0.0012314391979104883, 'lgbmr__num_leaves': 13, 'lgbmr__colsample_bytree': 0.9920040005884901, 'lgbmr__subsample': 0.6013906896641126, 'lgbmr__subsample_freq': 2, 'lgbmr__min_child_samples': 0}. Best is trial 63 with value: -9.418766279450413.
best_params = {'lgbmr__reg_alpha': 0.019197753824492524, 'lgbmr__reg_lambda': 0.0012922239440782994, 'lgbmr__num_leaves': 13, 'lgbmr__colsample_bytree': 0.9719197851402581, 'lgbmr__subsample': 0.5944071346631961, 'lgbmr__subsample_freq': 2, 'lgbmr__min_child_samples': 0}
score after tuning = -9.418766279450413
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
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしでスコアの上昇履歴を表示
オプション引数を指定しないとき、[デフォルトの引数]()を使用してOptunaでチューニングします
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でスコアの上昇履歴を表示 ######
tuning.plot_search_history()
```
実行結果
![image](https://user-images.githubusercontent.com/59557625/131160736-4235e9ef-a733-42b0-9e0b-f9ab84542a1d.png)

#### 横軸に時間を指定してスコアの上昇履歴を表示
`x_axis`引数='time'と指定する事で、横軸を試行数 → 時間に変更する事ができます。

```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### 時間を横軸に指定してスコアの上昇履歴を表示 ######
tuning.plot_search_history(x_axis='time')
```
実行結果
![image](https://user-images.githubusercontent.com/59557625/131169898-8876104b-0b66-4491-afdd-e7e3853db71b.png)

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L135)をご参照ください

<br>
<br>

## get_search_historyメソッド
チューニング進行に伴うスコアの上昇履歴 (plot_search_historyと同内容)をPandas DataFrameで取得します。

事前に、grid_search_tuning(), random_search_tuning(), bayes_opt_tuning(), optuna_tuning()いずれかを実行する必要があります

引数はありません

### 実行例

```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でスコアの上昇履歴を取得 ######
df_history = tuning.get_search_history()

df_history
```
実行結果
```
reg_alpha	reg_lambda	num_leaves	colsample_bytree	subsample	subsample_freq	min_child_samples	test_score	raw_trial_time	max_score	raw_total_time	total_time
0	0.001329	0.071145	37	0.759195	0.493611	1	2	-11.452939	1.599340	-11.452939	1.599340	1.626420
...	...	...	
199	0.001079	0.000451	42	0.945653	0.618689	1	1	-10.511359	1.620776	-9.616610	301.749652	306.858965
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
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしで探索履歴をプロット
オプション引数を指定しないとき、[デフォルトの引数]()を使用して探索履歴をプロットします
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数でスコアの上昇履歴を表示 ######
tuning.plot_search_map()
```
実行結果
![image](https://user-images.githubusercontent.com/59557625/131164885-2ae5288c-0a80-468d-8bb8-641eb52ba03b.png)

#### 図の枚数と軸のパラメータを指定してスコアの上昇履歴を表示
`pair_n`引数で、プロットする図の縦横枚数を指定する事ができます

`order`引数で、軸のパラメータをリストで指定する事ができます。リスト指定順に、グラフ横軸 → グラフ縦軸 → 全体縦軸 → 全体横軸の順番でプロットされます

```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### 時間を横軸に指定してスコアの上昇履歴を表示 ######
tuning.plot_search_map(pair_n=6, 
                       order=['min_child_samples', 'colsample_bytree', 'subsample', 'num_leaves'])
```
実行結果
![image](https://user-images.githubusercontent.com/59557625/131168816-b297e298-99fd-43c1-96b9-a2673b13aaaa.png)

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
コードは[こちらにもアップロードしています]()
#### オプション引数指定なしで学習曲線をプロット
オプション引数を指定しないとき、[デフォルトの引数]()を使用して学習曲線をプロットします
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
best_params, best_score = tuning.optuna_tuning()  # Optunaチューニング
###### デフォルト引数で学習曲線を表示 ######
tuning.plot_best_learning_curve()
```
実行結果
![image](https://user-images.githubusercontent.com/59557625/131206591-6ec6b08b-66d2-43a1-a138-519d62ac7a9f.png)

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L215)をご参照ください

<br>
<br>

# プロパティ一覧


# MLFlowの活用
