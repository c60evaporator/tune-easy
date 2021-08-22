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
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
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
※詳細は[チューニング手順]()の項を参照ください

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
[こちらの記事の方法]()をベースとして、パラメータチューニングを実施します。
Scikit-LearnのAPIに対応した学習器が対象となります。

## 0.1. データの読込＆前処理
使用するデータを読み込み、特徴量選択等の前処理を実行します。
詳細は

### 実行例
ボストン住宅価格データセットを読み込み、特徴量 (説明変数)を選択します
```python
from sklearn.datasets import load_boston
import pandas as pd

USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 選択した5つの説明変数
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values
y = load_boston().target  # 目的変数
```

※選択した5特徴量は、ランダムフォレストのRFE (再帰的特徴量削減)により選定
```python
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

X = load_boston().data
y = load_boston().target
selector = RFE(RandomForestRegressor(random_state=42), n_features_to_select=5)
selector.fit(X, y)
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

デフォルト(各メソッドのscoring引数を指定しないとき)では、以下の指標を使用します
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
[plot_first_validation_curve()メソッド]()で検証曲線をプロットし、[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#22-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E7%A8%AE%E9%A1%9E%E3%81%A8%E6%8E%A2%E7%B4%A2%E7%AF%84%E5%9B%B2%E3%81%AE%E9%81%B8%E6%8A%9E)パラメータ探索範囲を選択します

事前に[4.クロスバリデーション手法の選択]()を実施し、cv引数に指定する事が望ましいです

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
![image](https://user-images.githubusercontent.com/59557625/130344662-f059ca05-f44d-4003-b995-1530f205b302.png)
下図のように検証曲線から過学習にも未学習にもなりすぎていない範囲を抽出し、探索範囲とすることが望ましいです

### 3. 探索法を選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#%E6%89%8B%E9%A0%8634-%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E9%81%B8%E6%8A%9E%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)パラメータの探索法を選択します。
以下の4種類の探索法から使用したい手法を選び、対応したメソッドを選択します

|探索法|使用するメソッド|
|---|---|
|グリッドサーチ|[grid_search_tuning()]()|
|ランダムサーチ|[random_search_tuning()]()|
|ベイズ最適化 (BayesianOptimization)|[bayes_opt_tuning()]()|
|ベイズ最適化 (Optuna)|[optuna_tuning()]()|

### 4.1. クロスバリデーション手法を選択
[こちらを参考に](https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359#24-%E3%82%AF%E3%83%AD%E3%82%B9%E3%83%90%E3%83%AA%E3%83%87%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3)クロスバリデーションの手法を選択します。

### 実行例
クロスバリデーションに5分割KFoldを指定
```python
CV = KFold(n_splits=5, shuffle=True, random_state=42)
```

### 4.2 チューニング実行
[3.で選択したチューニング用メソッド]()に対し、

・[1.で選択した評価指標]()をscoring引数に
・[2.で選択したチューニング範囲]()をtuning_params引数に
・[4.1で選択したクロスバリデーション手法]()をcv引数に

指定し、実行します

### 実行例
Optunaでチューニング
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

tuning.optuna_tuning(scoring=SCORING,
                     tuning_params=TUNING_PARAMS)

```


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
上記クラスは、以下のように初期化(__init__()メソッド)します

### 引数
初期化の引数は以下のようになります
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|X|必須|np.ndarray|-|説明変数データ (2次元のndarray)|
|y|必須|np.ndarray|-|目的変数データ (1次元or2次元のndarray)|
|x_colnames|必須|list[str]|-|説明変数のフィールド名のリスト|
|y_colname|オプション|str|None|目的変数のフィールド名|
|cv_group|オプション|str|None|GroupKFold、LeaveOneGroupOutのグルーピング対象データ|
|eval_data_source|オプション|{'all', 'valid', 'train'}|'all'|eval_setの指定方法, 'all'ならeval_set =[(self.X, self.y)] (XGBoost, LightGBMのみ有効)|

### 実行例
#### 引数指定なし
LightGBM回帰におけるクラス初期化実行例
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# データセット読込
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
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
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
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
|estimator|必須|pd.DataFrame|-|入力データ|
|validation_curve_params|オプション|dict[str, float]|※後述|色分けに指定するカラム名 (Noneなら色分けなし)|
|palette|オプション|str|None|hueによる色分け用の[カラーパレット](https://matplotlib.org/stable/tutorials/colors/colormaps.html)|
|vars|オプション|list[str]|None|グラフ化するカラム名 (Noneなら全ての数値型＆Boolean型の列を使用)|
|lowerkind|オプション|str|'boxscatter'|左下に表示するグラフ種類 ('boxscatter', 'scatter', or 'reg')|
|diag_kind|オプション|str|'kde'|対角に表示するグラフ種類 ('kde' or 'hist')|
|markers|オプション|str or list[str]|None|hueで色分けしたデータの散布図プロット形状|
|height|オプション|float|2.5|グラフ1個の高さ|
|aspect|オプション|float|1|グラフ1個の縦横比|
|dropna|オプション|bool|True|[seaborn.PairGridのdropna引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|lower_kws|オプション|dict|{}|[seaborn.PairGrid.map_lowerの引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|diag_kws|オプション|dict|{}|[seaborn.PairGrid.map_diag引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|
|grid_kws|オプション|dict|{}|[seaborn.PairGridの上記以外の引数](https://seaborn.pydata.org/generated/seaborn.PairGrid.html?highlight=pairgrid#seaborn.PairGrid)|

### 実行例
LightGBM回帰における実行例 (引数指定なし)
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = boston[USE_EXPLANATORY].values
y = load_boston().target
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス初期化
###### 範囲を定めて検証曲線プロット ######
tuning.plot_first_validation_curve()
```
![image](https://user-images.githubusercontent.com/59557625/115889860-4e8bde80-a48f-11eb-826a-cd3c79556a42.png)

LightGBM回帰における引数指定例
```python
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # 説明変数
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = boston[USE_EXPLANATORY].values
y = load_boston().target
# チューニング実行
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)  # チューニング用クラス
tuning.plot_first_validation_curve()  # 範囲を定めて検証曲線をプロット
```

<br>

# プロパティ一覧
