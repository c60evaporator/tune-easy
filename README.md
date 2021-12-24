# muscle-tuning
[![python](https://img.shields.io/pypi/pyversions/muscle-tuning)](https://www.python.org/)
[![pypi](https://img.shields.io/pypi/v/muscle-tuning?color=blue)](https://pypi.org/project/muscle-tuning/)
[![license](https://img.shields.io/pypi/l/muscle-tuning?color=blue)](https://github.com/c60evaporator/muscle-tuning/blob/master/LICENSE)

**A hyperparameter tuning tool, easy to use even if your brain is made of muscle**

This documentation is Japanese language version.
**[English version is here](https://github.com/c60evaporator/muscle-tuning/blob/master/README.rst)**

**[API reference is here](https://c60evaporator.github.io/muscle-tuning/)**

<br>

# 使用例
## 一括チューニング
複数の機械学習アルゴリズムを一括チューニングして比較できます

```python
from muscle_tuning import MuscleTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores
```

<img width="320" src="https://user-images.githubusercontent.com/59557625/140383755-bca64ab3-1593-47ef-8401-affcd0b20a0a.png">

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png">


## 詳細チューニング
1種類の機械学習アルゴリズムのパラメータを詳細にチューニング・可視化できます

```python
from muscle_tuning import LGBMClassifierTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング実行と結果の可視化 ######
tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY)  # チューニング用クラス
tuning.plot_first_validation_curve(cv=2)  # 範囲を定めて検証曲線をプロット
tuning.optuna_tuning(cv=2)  # Optunaによるチューニング実行
tuning.plot_search_history()  # スコアの上昇履歴を可視化
tuning.plot_search_map()  # 探索点と評価指標を可視化
tuning.plot_best_learning_curve()  # 学習曲線の可視化
tuning.plot_best_validation_curve()  # 検証曲線の可視化
```
<img width="320" src="https://user-images.githubusercontent.com/59557625/145702586-8b341344-625c-46b3-a9ee-89cb592b1800.png">

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702594-cc4b2194-2ed0-40b0-8a83-94ebd8162818.png">

<img width="320" src="https://user-images.githubusercontent.com/59557625/145702643-70e3b1f2-66aa-4619-9703-57402b3669aa.png">


### MLflowによるチューニング履歴の記録

```python
from muscle_tuning import MuscleTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                             mlflow_logging=True)  # MLflowによる記録を指定
```

<img width="640" src="https://user-images.githubusercontent.com/59557625/147270240-f779cf1f-b216-42a2-8156-37169511ec3e.png">

<br>

# インストール方法
```
$ pip install muscle-tuning
```

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

# サポート
バグ等は[Issues](https://github.com/c60evaporator/muscle-tuning/issues)で報告してください。

機能追加の要望（対応する学習器の追加etc.）もIssuesまでお願いします。

<br>

# 使用法
以下の2種類のチューニング法のいずれかを選び、使用手順およびAPI仕様をリンク先から参照してください

|方法|クラス名|用途|使用手順リンク|API仕様リンク|
|---|---|---|---|---|
|一括チューニング|[MuscleTuning](https://github.com/c60evaporator/muscle-tuning#一括チューニング用クラス)|複数の機械学習アルゴリズムを一括チューニングして比較|[使用手順](https://github.com/c60evaporator/muscle-tuning#%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E6%89%8B%E9%A0%86-muscle_brain_tuning)|[API仕様]()|
|詳細チューニング|[学習器の種類毎に異なる](https://github.com/c60evaporator/muscle-tuning#詳細チューニング用クラス)|1種類の機械学習アルゴリズムのパラメータを詳細にチューニング|[使用手順](https://github.com/c60evaporator/muscle-tuning#%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E6%89%8B%E9%A0%86-%E8%A9%B3%E7%B4%B0%E3%83%81%E3%83%A5%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0)|[API仕様]()|

<br>

# クラス一覧
本ライブラリは以下のクラスからなります。用途に応じて使い分けてください

## 一括チューニング用クラス
複数の機械学習アルゴリズムを一括チューニングして比較したい際に使用するクラスです

|クラス名|パッケージ名|概要|API仕様(日本語)|API仕様(英語)|
|---|---|---|---|---|
|MuscleTuning|muscle_tuning.py|複数の機械学習アルゴリズムでのチューニングを一括実行、結果をグラフ表示|[API仕様]()|[API Reference]()|

<br>

## 詳細チューニング用クラス
1種類の機械学習アルゴリズムのパラメータを詳細にチューニングしたい際に使用するクラスです。
全てベースクラスである[muscle_tuning.param_tuning.ParamTuning]()クラスを継承しています。
メソッドのAPIリファレンスを見たい際には、[ベースクラスのAPIリファレンス](https://c60evaporator.github.io/muscle-tuning/param_tuning.html#muscle_tuning.param_tuning.ParamTuning)を参照ください

### - 分類

|クラス名|パッケージ名|概要|API仕様(日本語)|API仕様(英語)|
|---|---|---|---|---|
|LGBMClassifierTuning|lgbm_tuning.py|LightGBM分類のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|XGBClassifierTuning|xgb_tuning.py|XGBoost分類のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|SVMClassifierTuning|svm_tuning.py|サポートベクターマシン分類のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|RFClassifierTuning|rf_tuning.py|ランダムフォレスト分類のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|LogisticRegressionTuning|logisticregression_tuning.py|ロジスティック回帰分類のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|

### - 回帰

|クラス名|パッケージ名|概要|API仕様(日本語)|API仕様(英語)|
|---|---|---|---|---|
|LGBMRegressorTuning|lgbm_tuning.py|LightGBM回帰のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|XGBRegressorTuning|xgb_tuning.py|XGBoost回帰のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|SVMRegressorTuning|svm_tuning.py|サポートベクター回帰のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|RFRegressorTuning|rf_tuning.py|ランダムフォレスト回帰のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
|ElasticNetTuning|elasticnet_tuning.py|ElasticNet回帰のパラメータチューニング用クラス|[API仕様]()|[API Reference]()|
