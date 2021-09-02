# %%
from muscle_tuning import MuscleTuning
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import seaborn as sns

# Load dataset
# USE_EXPLANATORY = ['NOX', 'RM', 'DIS', 'LSTAT']
# df_boston = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
# X = df_boston[USE_EXPLANATORY].values
# y = load_boston().target
# df_boston['price'] = y

iris = sns.load_dataset("iris")
#iris = iris[iris['species'] != 'setosa']
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values

kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)
kinnikun.df_scores

# %% 評価指標算出テスト
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']
OBJECTIVE_VARIALBLE = 'species'  # 目的変数
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # 説明変数
y = iris[OBJECTIVE_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
le = LabelEncoder()
le.fit(y)
y_int = le.transform(y)
estimator = RandomForestClassifier()
scorings = {'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1_macro': 'f1_macro',
            'neg_log_loss': 'neg_log_loss',
            'roc_auc_ovr': 'roc_auc_ovr'}
scores = cross_validate(estimator, X, y_int,
                        scoring=scorings,
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        )
scores

# 結論
# precision, recall, f1は事前にLabelEncoderしないと算出できない
# 多クラス分類のprecision, recallは文字指定ではダメ、make_scorerメソッドでaverage引数に'binary'以外を指定する必要がある
# make_scorerメソッドはListではcross_validateに渡せないので、dict指定する必要あり


# %%
