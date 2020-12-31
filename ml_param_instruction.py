# %% 1-2) 標高と気圧で線形回帰
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import r2_score, accuracy_score
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.plotting import plot_decision_regions

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

# %% 1-2-A) 標高と気温で線形回帰
X = df_temp[['altitude']].values  # 説明変数(標高)
y = df_temp[['temperature']].values  # 目的変数(気温)
lr.fit(X, y)
plt.scatter(X, y, color = 'blue')
plt.plot(X, lr.predict(X), color = 'red')
plt.xlabel('altitude [m]')
plt.ylabel('temperature [°C]')
plt.text(1000, 0, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示

# %% 1-2-A) 緯度と気温で線形回帰
X = df_temp[['latitude']].values  # 説明変数(緯度)
y = df_temp[['temperature']].values  # 目的変数(気温)
lr.fit(X, y)
plt.scatter(X, y, color = 'blue')
plt.plot(X, lr.predict(X), color = 'red')
plt.xlabel('latitude [°]')
plt.ylabel('temperature [°C]')
plt.text(35, 10, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示

# %% 1-2-A) 予測値と実測値
sns.regplot(lr.predict(X), y, ci=0, scatter_kws={'color':'blue'})  # 目的変数の予測値と実測値をプロット
plt.xlabel('pred_value [°C]')  # 予測値
plt.ylabel('real_value [°C]')  # 実測値
plt.text(0, -10, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示

# %% 1-2-A) 2次元説明変数を3次元プロット
X = df_temp[['altitude', 'latitude']].values  # 説明変数(標高+緯度)
y = df_temp[['temperature']].values  # 目的変数(気温)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X[:, 0], X[:, 1], y)
ax.set_xlabel('altitude [m]')
ax.set_ylabel('latitude [°]')
ax.set_zlabel('temperature [°C]')

# %% 1-2-A) 予測値と実測値
lr.fit(X, y)  # 線形回帰実施
sns.regplot(lr.predict(X), y, ci=0, scatter_kws={'color':'blue'})  # 目的変数の予測値と実測値をプロット
plt.xlabel('pred_value [°C]')
plt.ylabel('real_value [°C]')
plt.text(0, -10, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示

# %% 1-2-B) 動物の身長と体重
df_animal = pd.read_csv(f'./animal_size.csv')
X = df_animal[['body_length']].values  # 説明変数(体長)
y = df_animal[['weight']].values  # 目的変数(体重)
plt.scatter(X, y, color = 'blue')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.xlabel('body_length [cm]')
plt.ylabel('weight [kg]')

# %% 1-2-B) 線形回帰
df_animal = df_animal[df_animal['name'] != 'Giraffe']  # キリンを除外
df_animal = df_animal.sort_values('body_length')  # 表示用に体長でソート
X = df_animal[['body_length']].values  # 説明変数(体長)
y = df_animal[['weight']].values  # 目的変数(体重)
lr.fit(X, y)  # 線形回帰実施
plt.scatter(X, y, color = 'blue')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.plot(X, lr.predict(X), color = 'red')
plt.xlabel('body_length [cm]')
plt.ylabel('weight [kg]')
plt.text(350, 1000, f'r2={r2_score(y, lr.predict(X))}')  # R2乗値を表示

# %% 1-2-B) 3次式で回帰
def cubic_fit(x, a):  # 回帰用方程式
    Y = a * x **3
    return Y
popt, pcov = curve_fit(cubic_fit, X[:,0], y[:,0])  # 最小二乗法でフィッティング
plt.scatter(X, y, color = 'blue')  # 説明変数と目的変数のデータ点の散布図をプロット
pred_y = cubic_fit(X, popt[0])  # 回帰線の作成
X_add = np.sort(np.vstack((X, np.array([[370],[500],[550],[600]]))), axis=0)  # 線が滑らかになるよう、プロット用にデータ補完
pred_y_add = cubic_fit(X_add, popt[0])  # 回帰線の作成(プロット用)
plt.plot(X_add, pred_y_add, color = 'red')  # 回帰線のプロット
plt.xlabel('body_length [cm]')
plt.ylabel('weight [kg]')
plt.text(400, 1000, f'r2={r2_score(y, pred_y)}')  # R2乗値を表示

# %% 1-3) NBA、NFL選手(LB)の身長体重
df_athelete = pd.read_csv(f'./nba_nfl_1.csv')  # データ読込
sns.scatterplot(x='height', y='weight', data=df_athelete, hue='league')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.xlabel('height [cm]')
plt.ylabel('weight [kg]')

# %% 1-3) 線形SVM
def label_str_to_int(y):  # 目的変数をstr型→int型に変換(plot_decision_regions用)
    label_names = list(dict.fromkeys(y[:, 0]))
    label_dict = dict(zip(label_names, range(len(label_names))))
    y_int=np.vectorize(lambda x: label_dict[x])(y)
    return y_int
def legend_int_to_str(ax, y):  # 凡例をint型→str型に変更(plot_decision_regions用)
    hans, labs = ax.get_legend_handles_labels()
    ax.legend(handles=hans, labels=list(dict.fromkeys(y[:, 0])))

X = df_athelete[['height','weight']].values  # 説明変数(身長、体重)
y = df_athelete[['league']].values  # 目的変数(種目)
y_int = label_str_to_int(y)  # 目的変数をint型に変換
model = SVC(kernel='linear')  # 線形SVMを定義
model.fit(X, y_int)  # SVM学習を実行
ax = plot_decision_regions(X, y_int[:, 0], clf=model) #識別面を可視化
plt.xlabel('height [cm]')  # x軸のラベル
plt.ylabel('weight [kg]')  # y軸のラベル
legend_int_to_str(ax, y)  # 凡例をint型→str型に変更

# %% 1-3-B) NBA、NFL選手(OF)の身長体重
df_athelete = pd.read_csv(f'./nba_nfl_2.csv')
sns.scatterplot(x='height', y='weight', data=df_athelete, hue='league')  # 説明変数と目的変数のデータ点の散布図をプロット
plt.xlabel('height [cm]')  # x軸のラベル
plt.ylabel('weight [kg]')  # y軸のラベル
# %% 1-3-B) RBFカーネルのgammaを変更
X = df_athelete[['height','weight']].values  # 説明変数(身長、体重)
y = df_athelete[['league']].values  # 目的変数(種目)
y_int = label_str_to_int(y)
for gamma in [0.1, 0.01, 0.001]:  # gammaを変えてループ
    model = SVC(kernel='rbf', gamma=gamma)  # RBFカーネルのSVMをgammaを変えて定義
    model.fit(X, y_int)  # SVM学習を実行
    ax = plot_decision_regions(X, y_int[:, 0], clf=model)
    plt.xlabel('height [cm]')
    plt.ylabel('weight [kg]')
    legend_int_to_str(ax, y)
    plt.text(175, 140, f'gamma={model.gamma}, C={model.C}')  # gammaとCを表示
    plt.show()
# %% 1-3-C) Cを変更
for C in [10, 1, 0.1]:  # gammaを変えてループ
    model = SVC(kernel='rbf', gamma=0.01, C=C)  # RBFカーネルのSVMをCを変えて定義
    model.fit(X, y_int)  # SVM学習を実行
    ax = plot_decision_regions(X, y_int[:, 0], clf=model) 
    plt.xlabel('height [cm]')
    plt.ylabel('weight [kg]')
    legend_int_to_str(ax, y)
    plt.text(175, 140, f'gamma={model.gamma}, C={model.C}')  # gammaとCを表示
    plt.show()
# %% gammaと
