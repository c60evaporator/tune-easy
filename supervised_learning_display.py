# %% 標高と気圧で線形回帰
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

df_temp = pd.read_csv(f'./temp_pressure.csv')
lr = LinearRegression()  # 線形回帰用クラス
X = df_temp[['altitude']].values  # 説明変数(標高)
y = df_temp[['pressure']].values  # 目的変数(気圧)
lr.fit(X, y)  # 線形回帰実施
plt.scatter(X, y, color = 'blue')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red')
plt.xlabel('altitude [m]')  # x軸のラベル
plt.ylabel('pressure [hPa]')  # y軸のラベル
plt.text(1000, 700, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示
# %% 標高と気温で線形回帰
X = df_temp[['altitude']].values  # 説明変数(標高)
y = df_temp[['temperature']].values  # 目的変数(気温)
lr.fit(X, y)  # 線形回帰実施
plt.scatter(X, y, color = 'blue')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red')
plt.xlabel('altitude [m]')  # x軸のラベル
plt.ylabel('temperature [°C]')  # y軸のラベル
plt.text(1000, 0, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示
# %% 緯度と気温で線形回帰
X = df_temp[['latitude']].values  # 説明変数(緯度)
y = df_temp[['temperature']].values  # 目的変数(気温)
lr.fit(X, y)  # 線形回帰実施
plt.scatter(X, y, color = 'blue')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red')
plt.xlabel('latitude [°]')  # x軸のラベル
plt.ylabel('temperature [°C]')  # y軸のラベル
plt.text(35, 10, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示

# %% 予測値と実測値
import seaborn as sns
sns.regplot(lr.predict(X), y, ci=0, scatter_kws={'color':'blue'})  # 目的変数の予測値と実測値をプロット
plt.xlabel('pred_value [°C]')  # 予測値
plt.ylabel('real_value [°C]')  # 実測値
plt.text(0, -10, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示
# %% 2次元
X = df_temp[['altitude', 'latitude']].values  # 説明変数(標高+緯度)
y = df_temp[['temperature']].values  # 目的変数(気温)
lr.fit(X, y)  # 線形回帰実施
sns.regplot(lr.predict(X), y, ci=0, scatter_kws={'color':'blue'})  # 目的変数の予測値と実測値をプロット
plt.xlabel('pred_value [°C]')  # 予測値
plt.ylabel('real_value [°C]')  # 実測値
plt.text(0, -10, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示
# %%
