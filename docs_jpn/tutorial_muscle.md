# 一括チューニング MuscleTuningクラス 使用手順
以下の手順で、複数の機械学習アルゴリズム（学習器）でパラメータチューニングを一括実行し、結果をグラフ表示できます

1. [`MuscleTuning`]()クラスのインスタンスを作成
2. [`muscle_brain_tuning()`]()メソッドを実行

チューニング自体は2行で終わります。カンタンですね！

<br>

以下の3つのユースケースに分けてサンプルコードを解説します

|ユースケース|リンク|
|---|---|
|分類タスク|[分類タスクでの使用例](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#分類タスクでの使用例)|
|回帰タスク|[分類タスクでの使用例](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#回帰タスクでの使用例)|
|MLflowで結果記録|[MLflowによる結果のロギング](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#mlflowによる結果のロギング)|

<br>

## 分類タスクでの使用例
分類タスクでは、スコアの上昇履歴とチューニング前後のROC曲線を表示します

```python
from muscle_tuning import MuscleTuning
import seaborn as sns
# データセット読込
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # 2クラスに絞る
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# チューニング実行
kinnikun = MuscleTuning()  # 1. インスタンス作成
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)  # 2. チューニングメソッド実行
kinnikun.df_scores  # スコア一覧DataFrameを表示
```
チューニング完了後に、以下の情報が表示されます

#### ・スコアの上昇履歴

<img width="320px" src="https://user-images.githubusercontent.com/59557625/140383755-bca64ab3-1593-47ef-8401-affcd0b20a0a.png">

#### ・チューニング前のROC曲線

<img width="640px" src="https://user-images.githubusercontent.com/59557625/140382285-206752d5-3def-44e3-a2ca-fc0871a5f181.png">

#### ・チューニング後のROC曲線

<img width="640px" src="https://user-images.githubusercontent.com/59557625/140382175-a8261675-33ee-4a07-9a1b-074890d95ecd.png">

#### ・チューニング前後のスコア

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png">

#### ・チューニング後の機械学習モデル使用法

<img width="400" src="https://user-images.githubusercontent.com/59557625/145702328-fa3845d9-10fd-43b6-8593-0544294a5c93.png">

<br>

## 回帰タスクでの使用例
回帰タスクでは、スコアの上昇履歴とチューニング前後の予測値-実測値プロットを表示します

```python
import parent_import
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
# データセット読込
OBJECTIVE_VARIALBLE = 'price'  # 目的変数
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数多いので1000点にサンプリング
y = california_housing[OBJECTIVE_VARIALBLE].values
X = california_housing[USE_EXPLANATORY].values
###### チューニング一括実行 ######
kinnikun = MuscleTuning()  # 1. インスタンス作成
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)  # 2. チューニングメソッド実行
kinnikun.df_scores
```
チューニング完了後に、以下の情報が表示されます

#### ・スコアの上昇履歴

<img width="320px" src="https://user-images.githubusercontent.com/59557625/145703714-2a83b29d-a8cc-4bab-a28f-5895cadcd44d.png">

#### ・チューニング前の予測値-実測値プロット

<img width="640px" src="https://user-images.githubusercontent.com/59557625/145703802-5f73fb07-e5a9-44df-9db6-e53b6a61e1ea.png">

#### ・チューニング後の予測値-実測値プロット

<img width="640px" src="https://user-images.githubusercontent.com/59557625/145703814-bca7a2c5-2f78-4c2e-9ff8-7eae8b1fd839.png">

#### ・チューニング前後のスコア

<img width="420" src="https://user-images.githubusercontent.com/59557625/145704306-738cf2ee-86ae-4d7d-8909-b416a5ec9b7c.png">

#### ・チューニング後の機械学習モデル使用法

<img width="400" src="https://user-images.githubusercontent.com/59557625/145703822-81940d44-229d-484d-b73a-5720282bb3a5.png">

上図の`----The following is how to use the best estimator----`以降のコードを以下のようにコピペすれば、
チューニング後のモデルを再現することができます

（後述のMLflowでPickleでも保存可能）

```python
from sklearn.ensemble import RandomForestRegressor
NOT_OPT_PARAMS = {'random_state': 42}
BEST_PARAMS = {'n_estimators': 86, 'max_features': 2, 'max_depth': 11, 'min_samples_split': 5, 'min_samples_leaf': 6}
params = {}
params.update(NOT_OPT_PARAMS)
params.update(BEST_PARAMS)
estimator = RandomForestRegressor()
estimator.set_params(**params)
estimator.fit(X, y)
```

<br>

## MLflowによる結果のロギング
`mlflow_logging`引数指定で、[MLflow](https://mlflow.org/docs/latest/tracking.html)による結果ロギングが可能です。

```python
from muscle_tuning import MuscleTuning
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
OBJECTIVE_VARIALBLE = 'price'  # Objective variable name
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # Selected explanatory variables
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
        columns = np.append(fetch_california_housing().feature_names, OBJECTIVE_VARIALBLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # sampling from 20640 to 1000
y = california_housing[OBJECTIVE_VARIALBLE].values  # Explanatory variables
X = california_housing[USE_EXPLANATORY].values  # Objective variable
###### チューニング一括実行 ######
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                             mlflow_logging=True)  # MLflowによるロギング有効化
kinnikun.df_scores
```
上記コードを実行後にMLflow UIを起動すると、以下のような画面が表示されます。

<img width="794" src="https://user-images.githubusercontent.com/59557625/145711588-be0e393f-be7b-4833-b17a-05eecd6ad014.png">

ロギング結果は、親RUN（全学習器の比較結果）と子RUN（各学習器ごとのチューニング結果）に分けてネストして保存されます。

<img width="292" src="https://user-images.githubusercontent.com/59557625/145711846-3a445abf-4013-44ef-862f-9ace3839ffe5.png">

<br>

### - 親RUNの保存内容
親RUNには、以下のような情報が保存されます
<img width="681" src="https://user-images.githubusercontent.com/59557625/145712480-15fc8916-2d16-410c-9889-50d18414fbe5.png">

#### ・Parameters

[`muscle_brain_tuning()`]()メソッドの引数を記録します

（ただし、`tuning_algo`, `x_colnames`, `y_colname`引数はTags, `estimators`, `tuning_params`, `tuning_kws`引数はArtifactとして記録）

<img width="397" src="https://user-images.githubusercontent.com/59557625/145712569-c36ad9aa-8912-44c3-b93a-215eed726ec0.png">

#### ・Metrics

学習器ごとのチューニング後のスコアが記録されます

<img width="300" src="https://user-images.githubusercontent.com/59557625/145712733-5dfe8b9d-e78e-4591-bc4a-76b05674846b.png">

青色のリンク部分をクリックすると、チューニング時のスコア推移をグラフ表示することが可能です

（横軸は試行数）

<img width="681" src="https://user-images.githubusercontent.com/59557625/145712813-c5bb903e-daa8-4421-b03f-97e7d8f4ef2c.png">

#### ・Tags

[`muscle_brain_tuning()`]()メソッドの`tuning_algo`, `x_colnames`, `y_colname`引数を記録します。

<img width="300" src="https://user-images.githubusercontent.com/59557625/145712936-a417a5a5-0d57-488a-b4b0-ad1ea9ca0c8d.png">

#### ・Artifacts
以下の内容を記録します

|名称|内容|備考|
|---|---|---|
|arg-estimators.json|[`muscle_brain_tuning()`]()メソッドの`estimators`引数||
|arg-tuning_params.json|[`muscle_brain_tuning()`]()メソッドの`tuning_params`引数||
|arg-tuning_kws.json|[`muscle_brain_tuning()`]()メソッドの`tuning_kws`引数||
|score_history.png|[スコアの上昇履歴](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#スコアの上昇履歴-1)||
|pred_true_before.png|[チューニング前の予測値-実測値プロット](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#チューニング前の予測値-実測値プロット)|回帰タスクのみ|
|pred_true_after.png|[チューニング後の予測値-実測値プロット](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#チューニング後の予測値-実測値プロット)|回帰タスクのみ|
|roc_curve_before.png|[チューニング前のROC曲線](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#チューニング前のroc曲線)|分類タスクのみ|
|roc_curve_after.png|[チューニング後のROC曲線](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#チューニング後のroc曲線)|分類タスクのみ|
|score_result.csv|[チューニング前後のスコア](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#チューニング前後のスコア-1)||
|score_result_cv.csv|チューニング前後のスコア (クロスバリデーションごと)|下図参照|
|how_to_use_best_estimator.py|[チューニング後の機械学習モデル使用法](https://github.com/c60evaporator/muscle-tuning/blob/master/docs_jpn/tutorial_muscle.md#チューニング後の機械学習モデル使用法-1)||
・score_result_cv.csvの表示例

<img width="681" alt="スクリーンショット 2021-12-12 21 55 11" src="https://user-images.githubusercontent.com/59557625/145713542-57f3f10d-7548-49ed-bb4b-ecc8e5e15b0b.png">

<br>

### - 子RUNの保存内容
子RUNの保存内容は、[詳細チューニングにおける保存内容]()と同様です。
