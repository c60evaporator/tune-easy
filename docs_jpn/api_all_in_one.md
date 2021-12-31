# 一括チューニング API仕様
一括チューニング実施用クラスのAPI仕様を記載します（[English version](https://c60evaporator.github.io/tune-easy/tune_easy.html#all-in-one-tuning-class)）

チューニングの実行手順は[こちら](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md)を参照ください

[サンプルコードはこちらにアップロードしています](https://github.com/c60evaporator/tune-easy/tree/master/examples/all_in_one_tuning)

# クラス一覧

|クラス名|パッケージ名|概要|デフォルトパラメータのリンク|
|---|---|---|---|
|AllInOneTuning|[all_in_one_tuning.py](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/all_in_one_tuning.py)|一括チューニング用クラス|[リンク](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/all_in_one_tuning.py#L27)|

## クラス初期化
上記クラスは、以下のように初期化します（引数指定はありません）

```python
all_tuner = AllInOneTuning()
```

<br>

# メソッド一覧
上記クラスは、以下のようなメソッドを持ちます

|メソッド名|機能|
|---|---|
|[all_in_one_tuning]()|一括チューニングを実行|
|[print_estimator]()|チューニング後の最適モデルの実装方法をコマンドラインに表示|

## all_in_one_tuningメソッド
一括チューニングを実行します

※本メソッドは内部的には`learning_algos`引数で指定した複数の詳細チューニングクラスを実行しています。
[詳細チューニングクラスのAPI仕様](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md)も併せてご参照ください

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|X|必須|np.ndarray or list[str]|-|説明変数データ (2次元のndarray(`data`引数未指定時) or 列名のリスト(`data`引数指定時))|
|y|必須|np.ndarray or str|-|目的変数データ (1次元のndarray (`data`引数未指定時) or 列名(`data`引数指定時))|
|data|オプション|list[str]|None|説明変数のフィールド名のリスト (`X`, `y`に列名指定時は必須)|
|x_colnames|オプション|list[str]|None|説明変数のフィールド名のリスト (`X`にndarray指定時のみ有効)|
|cv_group|オプション|str|None|GroupKFold、LeaveOneGroupOutのグルーピング対象データ (1次元のndarray(`data`引数未指定時) or 列名(`data`引数指定時))|
|objective|オプション|'classification' or 'regression'|データから自動判定|タスクの指定 ('classification':分類, 'regression':回帰)|
|scoring|オプション|str|'rmse' in regression, 'logloss' in clasification|最適化で最大化する評価指標 (指定できるスコアは[こちら](https://github.com/c60evaporator/tune-easy/blob/master/tune_easy/all_in_one_tuning.py#L59)参考)|
|other_scores|オプション|str|[タスクごとに異なるOTHER_SCORES定数](https://c60evaporator.github.io/tune-easy/all_in_one_tuning.html#tune_easy.all_in_one_tuning.AllInOneTuning.OTHER_SCORES)|チューニング前後で算出して比較表示する評価指標(戻り値のDataFrameに格納)|
|learning_algos|オプション|list[str]|[タスクごとに異なるLEARNING_ALGOS定数](https://c60evaporator.github.io/tune-easy/all_in_one_tuning.html#tune_easy.all_in_one_tuning.AllInOneTuning.LEARNING_ALGOS)|使用する[詳細チューニングクラス](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#クラス一覧)(学習器の種類)をリスト指定|
|n_iter|オプション|int|[タスクごとに異なるN_ITER定数](https://c60evaporator.github.io/tune-easy/all_in_one_tuning.html#tune_easy.all_in_one_tuning.AllInOneTuning.N_ITER)|ベイズ最適化の試行数|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|tuning_algo|オプション|str|{'grid', 'random', 'bo', or 'optuna'}|チューニングアルゴリズム ('grid':グリッドサーチ, 'random':ランダムサーチ, 'bo':BayesianOptimization ,'optuna':Optuna)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|estimators|オプション|dict[str, estimator object implementing 'fit']|`learning_algos`で指定した[チューニングクラス](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#クラス一覧)のデフォルト値|チューニングに使用する学習器インスタンスをdict指定 (Key:`learning_algos`の指定名, Value:学習器インスタンス(sckit-learn API))|
|tuning_params|オプション|dict[str, dict[str, {list, tuple}]]|`learning_algos`で指定した[チューニングクラス](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_each.md#クラス一覧)のデフォルト値|チューニングに使用するパラメータ範囲をdict指定 (Key:`learning_algos`の指定名, Value:パラメータ範囲(`tuning_algo`により[指定法異なる](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#3-探索法を選択)))|
|mlflow_logging|オプション|bool|False|MLflowでの結果記録有無(True:MLflow記録あり, False:MLflow記録なし|
|mlflow_<br>tracking_uri|オプション|str|None|MLflowのTracking URI。[こちらを参照ください]()|
|mlflow_<br>artifact_location|オプション　　|str|None|MLflowのArtifact URI。[こちらを参照ください]()|
|mlflow_<br>experiment_name|オプション|str|None|MLflowのExperiment名。[こちらを参照ください]()|
|tuning_kws|オプション|dict[str, dict]|None|詳細チューニング用メソッドに渡す変数 (Key:`learning_algos`の指定名, Value:チューニング用メソッドに渡したい引数(`tuning_algo`により[指定法異なる](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#3-探索法を選択)))|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/tune-easy/tree/master/examples/all_in_one_tuning)
#### 2クラス分類でチューニング一括実行（オプション引数なし）
オプション引数を指定しないとき、[前述のデフォルト値](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#引数一覧)を使用してプロットします

```python
from tune_easy import AllInOneTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### 一括チューニング実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY)
all_tuner.df_scores
```
実行結果（プロットされる図は割愛 → [実行手順参照](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#分類タスクでの使用例)）

<img width="480" src="https://user-images.githubusercontent.com/59557625/146761738-ae77db76-d542-4e03-9a37-b1c3b1ccdd5a.png">

#### 2クラス分類でチューニング一括実行（オプション引数なし）
オプション引数の指定例を下記します

```python
from tune_easy import AllInOneTuning
import seaborn as sns
from sklearn.svm import SVC
from xgboost import XGBClassifier
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# 引数を指定
not_opt_params_svm = {'kernel': 'rbf'}
not_opt_params_xgb = {'objective': 'binary:logistic',
                      'random_state': 42,
                      'booster': 'gbtree',
                      'n_estimators': 100,
                      'use_label_encoder': False}
fit_params_xgb = {'verbose': 0,
                  'eval_metric': 'logloss'}
tuning_params_svm = {'gamma': (0.001, 1000),
                     'C': (0.001, 1000)
                     }
tuning_params_xgb = {'learning_rate': (0.05, 0.3),
                     'min_child_weight': (1, 10),
                     'max_depth': (2, 9),
                     'colsample_bytree': (0.2, 1.0),
                     'subsample': (0.2, 1.0)
                     }
###### 一括チューニング実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY,
                        objective='classification', 
                        scoring='auc_ovo',
                        other_scores=['accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_weighted', 'auc_ovr', 'auc_ovo'],
                        learning_algos=['svm', 'xgboost'], 
                        n_iter={'svm': 50,
                                'xgboost': 20},
                        cv=3, tuning_algo='optuna', seed=42,
                        estimators={'svm': SVC(),
                                        'xgboost': XGBClassifier()},
                        tuning_params={'svm': tuning_params_svm,
                                        'xgboost': tuning_params_xgb},
                        tuning_kws={'svm': {'not_opt_params': not_opt_params_svm},
                                        'xgboost': {'not_opt_params': not_opt_params_xgb,
                                                'fit_params': fit_params_xgb}}
                        )
all_tuner.df_scores
```
実行結果（プロットされる図は割愛）

<img width="540" src="https://user-images.githubusercontent.com/59557625/146782743-85bc5816-82d4-4db9-ab43-21e15f973607.png">

#### 多クラス分類でチューニング一括実行（オプション引数なし）
オプション引数を指定しないとき、[前述のデフォルト値](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#引数一覧)を使用してプロットします

```python
from tune_easy import AllInOneTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
###### 一括チューニング実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY)
all_tuner.df_scores
```
実行結果（プロットされる図は割愛）

<img width="480" src="https://user-images.githubusercontent.com/59557625/146782390-fcf71e83-0513-4b2b-8d99-6185b2a239b1.png">

#### 多クラス分類でチューニング一括実行（オプション引数あり）
オプション引数の指定例を下記します

```python
from tune_easy import AllInOneTuning
import seaborn as sns
from sklearn.svm import SVC
from xgboost import XGBClassifier
# データセット読込
iris = sns.load_dataset("iris")
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# 引数指定
not_opt_params_svm = {'kernel': 'rbf'}
not_opt_params_xgb = {'objective': 'multi:softmax',
                      'random_state': 42,
                      'booster': 'gbtree',
                      'n_estimators': 100,
                      'use_label_encoder': False}
fit_params_xgb = {'verbose': 0,
                  'eval_metric': 'mlogloss'}
tuning_params_svm = {'gamma': (0.001, 1000),
                     'C': (0.001, 1000)
                     }
tuning_params_xgb = {'learning_rate': (0.05, 0.3),
                     'min_child_weight': (1, 10),
                     'max_depth': (2, 9),
                     'colsample_bytree': (0.2, 1.0),
                     'subsample': (0.2, 1.0)
                     }
###### 一括チューニング実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY,
                        objective='classification',
                        scoring='auc_ovo',
                        other_scores=['accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_weighted', 'auc_ovr', 'auc_ovo'],
                        learning_algos=['svm', 'xgboost'], 
                        n_iter={'svm': 50,
                                'xgboost': 20},
                        cv=3, tuning_algo='optuna', seed=42,
                        estimators={'svm': SVC(),
                                        'xgboost': XGBClassifier()},
                        tuning_params={'svm': tuning_params_svm,
                                        'xgboost': tuning_params_xgb},
                        tuning_kws={'svm': {'not_opt_params': not_opt_params_svm},
                                        'xgboost': {'not_opt_params': not_opt_params_xgb,
                                                'fit_params': fit_params_xgb}}
                        )
all_tuner.df_scores
```
実行結果（プロットされる図は割愛）

<img width="480" src="https://user-images.githubusercontent.com/59557625/146783147-dbd3d1cb-d465-4cbc-b88c-8297e902304a.png">

#### 回帰でチューニング一括実行（オプション引数なし）
オプション引数を指定しないとき、[前述のデフォルト値](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#引数一覧)を使用してプロットします

```python
from tune_easy import AllInOneTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# データセット読込
TARGET_VARIALBLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Objective variable
# Run tuning
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
all_tuner.df_scores
```
実行結果（プロットされる図は割愛）

<img width="480" src="https://user-images.githubusercontent.com/59557625/146791019-c869507f-f9ae-4700-b524-f710b621b7d9.png">

#### 回帰でチューニング一括実行（オプション引数あり）
オプション引数の指定例を下記します

```python
from tune_easy import AllInOneTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from xgboost import XGBRegressor
# Load dataset
TARGET_VARIALBLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Objective variable
# Set arguments
not_opt_params_svr = {'kernel': 'rbf'}
not_opt_params_xgb = {'objective': 'reg:squarederror',
                      'random_state': 42,
                      'booster': 'gbtree',
                      'n_estimators': 100,
                      'use_label_encoder': False}
fit_params_xgb = {'verbose': 0,
                  'eval_metric': 'rmse'}
tuning_params_svr = {'gamma': (0.001, 1000),
                     'C': (0.001, 1000),
                     'epsilon': (0, 0.3)
                     }
tuning_params_xgb = {'learning_rate': (0.05, 0.3),
                     'min_child_weight': (1, 10),
                     'max_depth': (2, 9),
                     'colsample_bytree': (0.2, 1.0),
                     'subsample': (0.2, 1.0)
                     }
# Run tuning
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY,
                        objective='regression',
                        scoring='mae',
                        other_scores=['rmse', 'mae', 'mape', 'r2'],
                        learning_algos=['svr', 'xgboost'], 
                        n_iter={'svr': 50,
                                'xgboost': 20},
                        cv=3, tuning_algo='optuna', seed=42,
                        estimators={'svr': SVR(),
                                        'xgboost': XGBRegressor()},
                        tuning_params={'svr': tuning_params_svr,
                                        'xgboost': tuning_params_xgb},
                        tuning_kws={'svr': {'not_opt_params': not_opt_params_svr},
                                        'xgboost': {'not_opt_params': not_opt_params_xgb,
                                                'fit_params': fit_params_xgb}}
                        )
all_tuner.df_scores
```
実行結果（プロットされる図は割愛）

<img width="480" src="https://user-images.githubusercontent.com/59557625/146788783-22107653-f277-47d7-9ba7-e05487b9d961.png">

<br>
<br>

## print_estimatorメソッド
チューニング後の学習器の使用方法をコマンドラインにprintします

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|learner_name|必須|str|-|print対象の学習器名。`learning_algos`で指定した名称の中から選択|
|printed_name|必須|str|-|'The following is how to use the xxx'の`xxx`の部分の文字列|
|mlflow_logging|オプション|bool|False|MLflowでの結果記録有無(True:MLflow記録あり, False:MLflow記録なし|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/tune-easy/blob/master/examples/all_in_one_tuning/print_estimator.py)

```python
from tune_easy import AllInOneTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
TARGET_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# 一括チューニング実行 
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY)
###### Print estimator ######
all_tuner.print_estimator('randomforest', 'randomforest estimator')
```
実行結果（プロットされる図は割愛）

<img width="480" src="https://user-images.githubusercontent.com/59557625/147111845-bf4225c1-429c-46c5-ac6f-379b721fcab7.png">
