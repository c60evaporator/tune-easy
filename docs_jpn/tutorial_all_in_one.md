# 一括チューニング 実行手順
以下の手順で、複数の機械学習アルゴリズム（学習器）でパラメータチューニングを一括実行し、結果をグラフ表示できます

1. [`AllInOneTuning`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#クラス初期化)クラスのインスタンスを作成
2. [`all_in_one_tuning()`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#all_in_one_tuningメソッド)メソッドを実行

チューニング自体は2行で終わります。カンタンですね！

<br>

以下の3つのユースケースに分けてサンプルコードを解説します

|ユースケース|リンク|
|---|---|
|分類タスク|[分類タスクでの使用例](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#分類タスクでの使用例)|
|回帰タスク|[分類タスクでの使用例](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#回帰タスクでの使用例)|
|MLflowで結果記録|[MLflowによる結果のロギング](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#mlflowによる結果のロギング)|

<br>

## 分類タスクでの使用例
分類タスクでは、スコアの上昇履歴とチューニング前後のROC曲線を表示します

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
# チューニング実行
all_tuner = AllInOneTuning()  # 1. インスタンス作成
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)  # 2. チューニングメソッド実行
all_tuner.df_scores  # スコア一覧DataFrameを表示
```
チューニング完了後に、以下の情報が表示されます

#### ・スコアの上昇履歴

<img width="320px" src="https://user-images.githubusercontent.com/59557625/165905780-d153541a-6c74-4dc6-a37f-7d63151bf582.png">

#### ・チューニング前のROC曲線

<img width="640px" src="https://user-images.githubusercontent.com/59557625/165906414-7928c742-b4cf-49c4-9a9a-3f75af306114.png">

#### ・チューニング後のROC曲線

<img width="640px" src="https://user-images.githubusercontent.com/59557625/165906496-2adc4f51-c9c3-4a4e-960d-12175b799217.png">

#### ・チューニング前後のスコア

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png">

#### ・チューニング後の機械学習モデル使用法

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702328-fa3845d9-10fd-43b6-8593-0544294a5c93.png">

<br>

## 回帰タスクでの使用例
回帰タスクでは、スコアの上昇履歴とチューニング前後の予測値-実測値プロットを表示します

```python
import parent_import
from tune_easy import AllInOneTuning
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
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()  # 1. インスタンス作成
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)  # 2. チューニングメソッド実行
all_tuner.df_scores
```
チューニング完了後に、以下の情報が表示されます

#### ・スコアの上昇履歴

<img width="320px" src="https://user-images.githubusercontent.com/59557625/165920858-d74df476-d9c7-4359-9851-94e9e5339e9b.png">

#### ・チューニング前の予測値-実測値プロット

<img width="640px" src="https://user-images.githubusercontent.com/59557625/145703802-5f73fb07-e5a9-44df-9db6-e53b6a61e1ea.png">

#### ・チューニング後の予測値-実測値プロット

<img width="640px" src="https://user-images.githubusercontent.com/59557625/165921166-c40f7c44-02a4-4991-bd9b-643236f948d0.png">

#### ・チューニング前後のスコア

<img width="420" src="https://user-images.githubusercontent.com/59557625/165928968-2c802ac9-aa3b-4293-a86c-70226323c975.png">

#### ・チューニング後の機械学習モデル使用法

<img width="640" src="https://user-images.githubusercontent.com/59557625/165928913-1783ab4d-8faf-446f-ade8-bbc0f6f96f63.png">

上図の`----The following is how to use the best estimator----`以降のコードを以下のようにコピペすれば、
チューニング後のモデルを再現することができます

（後述のMLflowでPickleでも保存可能）

```python
from lightgbm import LGBMRegressor
NOT_OPT_PARAMS = {'objective': 'regression', 'random_state': 42, 'boosting_type': 'gbdt', 'n_estimators': 10000}
BEST_PARAMS = {'reg_alpha': 0.00013463201150608505, 'reg_lambda': 0.03832397165103329, 'num_leaves': 6, 'colsample_bytree': 0.6351581078709001, 'subsample': 0.5251370914241965, 'subsample_freq': 1, 'min_child_samples': 9}
params = {}
params.update(NOT_OPT_PARAMS)
params.update(BEST_PARAMS)
estimator = LGBMRegressor()
estimator.set_params(**params)
FIT_PARAMS = {'verbose': 0, 'early_stopping_rounds': 10, 'eval_metric': 'rmse', 'eval_set': [(X, y)]}
estimator.fit(X, y, FIT_PARAMS)
```

<br>

## MLflowによる結果のロギング
`mlflow_logging`引数指定で、[MLflow](https://mlflow.org/docs/latest/tracking.html)による結果ロギングが可能です。

```python
from tune_easy import AllInOneTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
TARGET_VARIABLE = 'price'  #  Target variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[TARGET_VARIABLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  #  Target variable
###### チューニング一括実行 ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                           mlflow_logging=True)  # MLflowによるロギング有効化
all_tuner.df_scores
```
チューニング完了後、ターミナルで以下のコマンドを打つと、MLflowのUI用Webサーバが立ち上がります。

```
mlflow ui
```
`mlflow_logging`以外のMLflow用引数(`mlflow_tracking_uri`, `mlflow_artifact_location`, `mlflow_experiment_name`)を指定していなければ、ローカルホストにUIが作成される（[こちらの記事](https://qiita.com/c60evaporator/items/e1fd57a0263a19b629d1#シナリオ1-mlflow-on-localhost)のシナリオ1に相当）ので、ブラウザに`http://127.0.0.1:5000`と打つと、以下のような画面が表示されます。

<img width="794" src="https://user-images.githubusercontent.com/59557625/145711588-be0e393f-be7b-4833-b17a-05eecd6ad014.png">

ロギング結果は、親RUN（全学習器の比較結果）と子RUN（各学習器ごとのチューニング結果）に分けてネストして保存されます。

<img width="292" src="https://user-images.githubusercontent.com/59557625/145711846-3a445abf-4013-44ef-862f-9ace3839ffe5.png">

各RUNの`Start Time`をクリックすると、保存されている情報の詳細を表示させることができます。詳細を後述します。
<br>

### - 親RUNの保存内容
親RUNには、以下のような情報が保存されます

<img width="681" src="https://user-images.githubusercontent.com/59557625/145712480-15fc8916-2d16-410c-9889-50d18414fbe5.png">

#### ・Parameters

[`all_in_one_tuning()`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#all_in_one_tuningメソッド)メソッドの引数を記録します

（ただし、`tuning_algo`, `x_colnames`, `y_colname`引数はTags, `estimators`, `tuning_params`, `tuning_kws`引数はArtifactとして記録）

<img width="397" src="https://user-images.githubusercontent.com/59557625/145712569-c36ad9aa-8912-44c3-b93a-215eed726ec0.png">

#### ・Metrics

学習器ごとのチューニング後のスコアが記録されます

<img width="300" src="https://user-images.githubusercontent.com/59557625/145712733-5dfe8b9d-e78e-4591-bc4a-76b05674846b.png">

青色のリンク部分をクリックすると、チューニング時のスコア推移をグラフ表示することが可能です

（横軸は試行数）

<img width="681" src="https://user-images.githubusercontent.com/59557625/145712813-c5bb903e-daa8-4421-b03f-97e7d8f4ef2c.png">

#### ・Tags

[`all_in_one_tuning()`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#all_in_one_tuningメソッド)メソッドの`tuning_algo`, `x_colnames`, `y_colname`引数を記録します。

<img width="300" src="https://user-images.githubusercontent.com/59557625/145712936-a417a5a5-0d57-488a-b4b0-ad1ea9ca0c8d.png">

#### ・Artifacts
以下の内容を記録します

|名称|内容|備考|
|---|---|---|
|arg-estimators.json|[`all_in_one_tuning()`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#all_in_one_tuningメソッド)メソッドの`estimators`引数||
|arg-tuning_params.json|[`all_in_one_tuning()`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#all_in_one_tuningメソッド)メソッドの`tuning_params`引数||
|arg-tuning_kws.json|[`all_in_one_tuning()`](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/api_all_in_one.md#all_in_one_tuningメソッド)メソッドの`tuning_kws`引数||
|score_history.png|[スコアの上昇履歴](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#スコアの上昇履歴-1)||
|pred_true_before.png|[チューニング前の予測値-実測値プロット](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#チューニング前の予測値-実測値プロット)|回帰タスクのみ|
|pred_true_after.png|[チューニング後の予測値-実測値プロット](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#チューニング後の予測値-実測値プロット)|回帰タスクのみ|
|roc_curve_before.png|[チューニング前のROC曲線](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#チューニング前のroc曲線)|分類タスクのみ|
|roc_curve_after.png|[チューニング後のROC曲線](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#チューニング後のroc曲線)|分類タスクのみ|
|score_result.csv|[チューニング前後のスコア](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#チューニング前後のスコア-1)||
|score_result_cv.csv|チューニング前後のスコア (クロスバリデーションごと)|下図参照|
|how_to_use_best_estimator.py|[チューニング後の機械学習モデル使用法](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_all_in_one.md#チューニング後の機械学習モデル使用法-1)||

・score_result_cv.csvの表示例

<img width="681" src="https://user-images.githubusercontent.com/59557625/145713542-57f3f10d-7548-49ed-bb4b-ecc8e5e15b0b.png">

<br>

### - 子RUNの保存内容
子RUNの保存内容は、[詳細チューニングにおける保存内容](https://github.com/c60evaporator/tune-easy/blob/master/docs_jpn/tutorial_each.md#mlflowによる結果ロギング)と同様です。
