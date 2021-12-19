# 詳細チューニング API仕様
詳細チューニング実施用クラスのAPI仕様を記載します（[English version](https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#all-in-one-tuning-class)）

チューニングの実行手順は[こちら](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md)を参照ください

[サンプルコードはこちらにアップロードしています](https://github.com/c60evaporator/muscle-tuning/tree/master/examples/muscle_brain_tuning)

# クラス一覧

|クラス名|パッケージ名|概要|デフォルトパラメータのリンク|
|---|---|---|---|
|MuscleTuning|[muscle_tuning.py](https://github.com/c60evaporator/muscle-tuning/blob/master/muscle_tuning/muscle_tuning.py)|一括チューニング用クラス|[リンク](https://github.com/c60evaporator/muscle-tuning/blob/master/muscle_tuning/muscle_tuning.py#L27)|

## クラス初期化
上記クラスは、以下のように初期化します（引数指定はありません）

```python
kinnikun = MuscleTuning()
```

<br>

# メソッド一覧
上記クラスは、以下のようなメソッドを持ちます

|メソッド名|機能|
|---|---|
|[muscle_brain_tuning]()|一括チューニングを実行|
|[print_estimator]()|チューニング後の最適モデルの実装方法をコマンドラインに表示|

## muscle_brain_tuningメソッド
一括チューニングを実行します

### 引数一覧
|引数名|必須引数orオプション|型|デフォルト値|内容|
|---|---|---|---|---|
|X|必須|np.ndarray or list[str]|-|説明変数データ (2次元のndarray(`data`引数未指定時) or 列名のリスト(`data`引数指定時))|
|y|必須|np.ndarray or str|-|目的変数データ (1次元のndarray (`data`引数未指定時) or 列名(`data`引数指定時))|
|data|オプション|list[str]|None|説明変数のフィールド名のリスト (`X`, `y`に列名指定時は必須)|
|x_colnames|オプション|list[str]|None|説明変数のフィールド名のリスト (`X`にndarray指定時のみ有効)|
|cv_group|オプション|str|None|GroupKFold、LeaveOneGroupOutのグルーピング対象データ (1次元のndarray(`data`引数未指定時) or 列名(`data`引数指定時))|
|objective|オプション|'classification' or 'regression'|データから自動判定|タスクの指定 ('classification':分類, 'regression':回帰)|
|scoring|オプション|str|'rmse' in regression, 'logloss' in clasification|最適化で最大化する評価指標 (指定できるスコアは[こちら](https://github.com/c60evaporator/muscle-tuning/blob/master/muscle_tuning/muscle_tuning.py#L59)参考)|
|other_scores|オプション|str|[タスクごとに異なるOTHER_SCORES定数](https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.OTHER_SCORES)|チューニング前後で算出して比較表示する評価指標(戻り値のDataFrameに格納)|
|learning_algos|オプション|list[str]|[タスクごとに異なるLEARNING_ALGOS定数](https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.LEARNING_ALGOS)|使用する[個別チューニングクラス](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/api_each.md#クラス一覧)(学習器の種類)をリスト指定|
|n_iter|オプション|int|[タスクごとに異なるN_ITER定数](https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.N_ITER)|ベイズ最適化の試行数|
|cv|オプション|int, cross-validation generator, or an iterable|5|クロスバリデーション分割法 (int入力時はKFoldで分割)|
|tuning_algo|オプション|str|{'grid', 'random', 'bo', or 'optuna'}|チューニングアルゴリズム ('grid':グリッドサーチ, 'random':ランダムサーチ, 'bo':BayesianOptimization ,'optuna':Optuna)|
|seed|オプション|int|42|乱数シード (学習器の`random_state`に適用、`cv`引数がint型のときKFoldの乱数シードにも指定)|
|estimators|オプション|dict[str, estimator object implementing 'fit']|`learning_algos`で指定した[チューニングクラス](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/api_each.md#クラス一覧)のデフォルト値|チューニングに使用する学習器インスタンスをdict指定 (Key:`learning_algos`の指定名, Value:学習器インスタンス(sckit-learn API))|
|tuning_params|オプション|dict[str, dict[str, {list, tuple}]]|`learning_algos`で指定した[チューニングクラス](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/api_each.md#クラス一覧)のデフォルト値|チューニングに使用するパラメータ範囲をdict指定 (Key:`learning_algos`の指定名, Value:パラメータ範囲(`tuning_algo`により[指定法異なる](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_each.md#3-探索法を選択)))|
|mlflow_logging|オプション|bool|False|MLflowでの結果記録有無(True:MLflow記録あり, False:MLflow記録なし|
|mlflow_<br>tracking_uri|オプション|str|None|MLflowのTracking URI。[こちらを参照ください]()|
|mlflow_<br>artifact_location|オプション　　|str|None|MLflowのArtifact URI。[こちらを参照ください]()|
|mlflow_<br>experiment_name|オプション|str|None|MLflowのExperiment名。[こちらを参照ください]()|
|tuning_kws|オプション|dict[str, dict]|None|個別チューニング用メソッドに渡す変数 (Key:`learning_algos`の指定名, Value:チューニング用メソッドに渡したい引数(`tuning_algo`により[指定法異なる](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_each.md#3-探索法を選択)))|

### 実行例
コードは[こちらにもアップロードしています](https://github.com/c60evaporator/muscle-tuning/tree/master/examples/muscle_brain_tuning)
#### 2クラス分類でチューニング一括実行（オプション引数なし）
オプション引数を指定しないとき、[前述のデフォルト値]()を使用してプロットします

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
###### 一括チューニング実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)
kinnikun.df_scores
```
実行結果

<img width="800px" src="https://user-images.githubusercontent.com/59557625/145722194-0791ecc7-6fad-4c7a-a02e-71ec82d0c6bd.png">

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L123)をご参照ください

#### 多クラス分類でチューニング一括実行（オプション引数なし）

その他の引数の使用法は、[こちらのサンプルコード](https://github.com/c60evaporator/param-tuning-utility/blob/master/examples/regression_original/example_lgbm_regression.py#L123)をご参照ください

#### 回帰でチューニング一括実行（オプション引数なし）


<br>



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
