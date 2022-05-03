# tune-easy
[![python](https://img.shields.io/pypi/pyversions/tune-easy)](https://www.python.org/)
[![pypi](https://img.shields.io/pypi/v/tune-easy?color=blue)](https://pypi.org/project/tune-easy/)
[![license](https://img.shields.io/pypi/l/tune-easy)](https://github.com/c60evaporator/tune-easy/blob/master/LICENSE)

**A hyperparameter tuning tool for Machine Learning, extremely easy to use.**

<img width="320" src="https://user-images.githubusercontent.com/59557625/165905780-d153541a-6c74-4dc6-a37f-7d63151bf582.png"><img width="320" src="https://user-images.githubusercontent.com/59557625/166449590-b1f1efaf-00c8-432e-92d7-994085c034c6.png">

<img width="540" src="https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png">

This documentation is Japanese language version.
**[English version is here](https://github.com/c60evaporator/tune-easy/blob/master/README.rst)**

**[API reference is here](https://c60evaporator.github.io/tune-easy/)**

<br>

# 使用例
## 一括チューニング
複数の機械学習アルゴリズムを一括チューニングして比較できます

```python
from tune_easy import AllInOneTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIABLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
all_tuner.df_scores
```

<img width="320" src="https://user-images.githubusercontent.com/59557625/165905780-d153541a-6c74-4dc6-a37f-7d63151bf582.png">

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png">


## 詳細チューニング
1種類の機械学習アルゴリズムのパラメータを詳細にチューニング・可視化できます

```python
from tune_easy import LGBMClassifierTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIABLE].values
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
from tune_easy import AllInOneTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIABLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIABLE].values
X = iris[USE_EXPLANATORY].values
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                             mlflow_logging=True)  # MLflowによる記録を指定
```

```shell
$ mlflow ui
```

<img width="800" src="https://user-images.githubusercontent.com/59557625/147270240-f779cf1f-b216-42a2-8156-37169511ec3e.png">

<br>

# インストール方法
```
$ pip install tune-easy
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
* MLflow >=1.17.0
* LightGBM >=3.3.2
* XGBoost >=1.4.2
* seaborn-analyzer>=0.2.11
<br>

# サポート
バグ等は[Issues](https://github.com/c60evaporator/tune-easy/issues)で報告してください。

機能追加の要望（対応する学習器の追加etc.）もIssuesまでお願いします。

<br>

# 使用法
以下の2種類のチューニング法のいずれかを選び、使用手順およびAPI仕様をリンク先から参照してください

|方法|クラス名|用途|使用手順リンク|API仕様リンク|
|---|---|---|---|---|
|一括チューニング|[AllInOneTuning](https://github.com/c60evaporator/tune-easy#一括チューニング用クラス)|複数の機械学習アルゴリズムを一括チューニングして比較|[使用手順](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md)|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md)|
|詳細チューニング|[学習器の種類毎に異なる](https://github.com/c60evaporator/tune-easy#詳細チューニング用クラス)|1種類の機械学習アルゴリズムのパラメータを詳細にチューニング|[使用手順](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md)|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|

<br>

# クラス一覧
本ライブラリは以下のクラスからなります。用途に応じて使い分けてください

## 一括チューニング用クラス
複数の機械学習アルゴリズムを一括チューニングして比較したい際に使用するクラスです

|クラス名|ファイル名|概要|API仕様(日本語)|API仕様(英語)|
|---|---|---|---|---|
|AllInOneTuning|[all_in_one_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/all_in_one_tuning.py)|複数の機械学習アルゴリズムでのチューニングを一括実行、結果をグラフ表示|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md)|[API Reference]()|

<br>

## 詳細チューニング用クラス
1種類の機械学習アルゴリズムのパラメータを詳細にチューニングしたい際に使用するクラスです。
全てベースクラスである[tune_easy.param_tuning.ParamTuning]()クラスを継承しています。
メソッドのAPIリファレンスを見たい際には、[ベースクラスのAPIリファレンス](https://c60evaporator.github.io/tune-easy/param_tuning.html#tune_easy.param_tuning.ParamTuning)を参照ください

### - 回帰

|クラス名|ファイル名|概要|API仕様(日本語)|API仕様(英語)|
|---|---|---|---|---|
|LGBMRegressorTuning|[lgbm_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/lgbm_tuning.py)|LightGBM回帰のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|XGBRegressorTuning|[xgb_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/xgb_tuning.py)|XGBoost回帰のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|SVMRegressorTuning|[svm_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/svm_tuning.py)|サポートベクター回帰のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|RFRegressorTuning|[rf_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/rf_tuning.py)|ランダムフォレスト回帰のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|ElasticNetTuning|[elasticnet_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/elasticnet_tuning.py)|ElasticNet回帰のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|

### - 分類

|クラス名|ファイル名|概要|API仕様(日本語)|API仕様(英語)|
|---|---|---|---|---|
|LGBMClassifierTuning|[lgbm_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/lgbm_tuning.py)|LightGBM分類のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|XGBClassifierTuning|[xgb_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/xgb_tuning.py)|XGBoost分類のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|SVMClassifierTuning|[svm_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/svm_tuning.py)|サポートベクターマシン分類のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|RFClassifierTuning|[rf_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/rf_tuning.py)|ランダムフォレスト分類のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
|LogisticRegressionTuning|[logisticregression_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/logisticregression_tuning.py)|ロジスティック回帰分類のパラメータチューニング用クラス|[API仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md)|[API Reference]()|
