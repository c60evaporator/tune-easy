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
boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = boston[USE_EXPLANATORY].values
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
※その他のクラスの使用法は[構成]()の項を参照ください

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

# 概要
[こちらの記事の方法]()をベースとして、パラメータチューニングを実施します。
Scikit-LearnのAPIに対応した学習器が対象となります。

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

<br>

## クラス初期化引数一覧
上記クラスは、以下のように初期化(__init__()メソッド)します

### 実行例
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

XGBoostにおける引数指定例
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
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY,  # 必須引数
                             y_colname=OBJECTIVE_VARIABLE,  # 目的変数のフィールド名 (大阪都構想の賛成率)
                             cv_group=df_reg['ward_after'].values,  # グルーピング対象データ (大阪都構想の区)
                             eval_data_source='valid')  # eval_setの指定方法 (検証用データを渡す)
```
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

<br>

### plot_first_validation_curveメソッド
範囲を定めて検証曲線をプロットし、パラメータ調整範囲の参考とします

#### 実行例
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

#### 引数一覧
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
<br>

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