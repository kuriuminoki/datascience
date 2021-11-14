import numpy as np
import pandas as pd
from pandas import plotting
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --主成分分析器
from sklearn.decomposition import PCA


# --主成分分析を行う
def pca_analysis(df):
    # --標準化 0列目はのぞく
    dfs = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    dfs.head()
    # --主成分分析の実行
    pca = PCA()
    pca.fit(dfs)
    # --データを主成分空間に写像
    feature = pca.transform(dfs)
    # 主成分得点
    pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]).head()

    # --featureは [データ数＊主成分の数] のベクトル
    # print(feature.shape)
    # print(feature)

    # --累積寄与率をプロット
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()

    return dfs, feature


# --第x主成分と第y主成分をプロット column_nameごとに色分け
def pca_plot(df, feature, x, y, column_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(feature[:, x - 1], feature[:, y - 1], alpha=0.8, c=list(df[column_name]))
    plt.grid()
    plt.xlabel("PC{}".format(x))
    plt.ylabel("PC{}".format(y))
    plt.legend()
    plt.show()