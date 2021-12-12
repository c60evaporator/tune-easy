## チューニング手順 (一括チューニング)
[`MuscleTuning`]()クラスの[`muscle_brain_tuning()`]()メソッドを実行するのみで、
複数の機械学習アルゴリズムでパラメータチューニングを一括実行し、結果をグラフ表示できます

### 分類タスクでの使用例
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
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores  # スコア一覧DataFrameを表示
```
チューニング完了後に、以下の情報が表示されます

**・スコアの上昇履歴**

<img width="320px" src="https://user-images.githubusercontent.com/59557625/140383755-bca64ab3-1593-47ef-8401-affcd0b20a0a.png">

**・チューニング前のROC曲線**

<img width="640px" src="https://user-images.githubusercontent.com/59557625/140382285-206752d5-3def-44e3-a2ca-fc0871a5f181.png">

**・チューニング後のROC曲線**

<img width="640px" src="https://user-images.githubusercontent.com/59557625/140382175-a8261675-33ee-4a07-9a1b-074890d95ecd.png">

**・チューニング前後のスコア**

<img width="480" src="https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png">

**・チューニング後の機械学習モデル使用法**

<img width="400" src="https://user-images.githubusercontent.com/59557625/145702328-fa3845d9-10fd-43b6-8593-0544294a5c93.png">


### 回帰タスク
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
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
kinnikun.df_scores
```
チューニング完了後に、以下の情報が表示されます

**・スコアの上昇履歴**

<img width="320px" src="https://user-images.githubusercontent.com/59557625/145703714-2a83b29d-a8cc-4bab-a28f-5895cadcd44d.png">

**・チューニング前の予測値-実測値プロット**

<img width="640px" src="https://user-images.githubusercontent.com/59557625/145703802-5f73fb07-e5a9-44df-9db6-e53b6a61e1ea.png">

**・チューニング後の予測値-実測値プロット**

<img width="640px" src="https://user-images.githubusercontent.com/59557625/145703814-bca7a2c5-2f78-4c2e-9ff8-7eae8b1fd839.png">

**・チューニング前後のスコア**

<img width="420" src="https://user-images.githubusercontent.com/59557625/145704306-738cf2ee-86ae-4d7d-8909-b416a5ec9b7c.png">

**・チューニング後の機械学習モデル使用法**

<img width="400" src="https://user-images.githubusercontent.com/59557625/145703822-81940d44-229d-484d-b73a-5720282bb3a5.png">