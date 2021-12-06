from load_save import save_data

import numpy as np
import pandas as pd
from pandas import plotting
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# --主成分分析器
from sklearn.decomposition import PCA


# --主成分分析を行う
def pca_analysis(df, focus, dir_name=""):
    # --色分け基準の列を削除
    df = df.drop(columns=focus)
    # --標準化 0列目はのぞく
    dfs = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    # dfs = df
    dfs.head()
    # --主成分分析の実行
    pca = PCA()
    pca.fit(dfs)
    # --データを主成分空間に写像
    feature = pca.transform(dfs)
    # --主成分得点
    sum = min(len(dfs.columns), len(dfs.index))
    result = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(sum)])
    # print(result)
    save_data(result, "result/pca/{}主成分得点.csv".format(dir_name))

    # --主成分負荷量を算出
    eigen_vector = pca.components_
    # --データフレームに変換
    weight = pd.DataFrame(eigen_vector,
                          columns=dfs.columns,
                          index=["主成分{}".format(x + 1) for x in range(sum)])
    save_data(weight, "result/pca/{}負荷.csv".format(dir_name))

    # --負荷をソートして保存
    if True:
        tweight = weight.T
        for i in range(1, 11):
            s = tweight["主成分{}".format(i)]
            s = s.sort_values().T
            save_data(s, "result/pca/{}主成分{}.csv".format(dir_name, i))
            plt.figure(figsize=(10, 8))
            plt.rcParams['font.family'] = 'Hiragino sans'
            plt.barh(s.index, s.values)
            plt.grid()
            plt.savefig("result/pca/{}主成分{}負荷.png".format(dir_name, i))
            plt.clf()

    # --累積寄与率をプロット
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.savefig("result/pca/{}累積寄与率.png".format(dir_name))
    #plt.show()
    plt.clf()

    return dfs, feature


# --第x主成分と第y主成分をプロット column_nameごとに色分け
def pca_plot(df, feature, x, y, column_name, dir_name=""):
    plt.figure(figsize=(6, 6))
    plt.scatter(feature[:, x - 1], feature[:, y - 1], alpha=0.8, c=list(df[column_name]))
    #plt.scatter(feature[:, x - 1], feature[:, y - 1], alpha=0.8, c=list(df[column_name]),
                #cmap='Blues', vmin=80.0, vmax=30.0)
    plt.colorbar()
    plt.grid()
    plt.xlabel("PC{}".format(x))
    plt.ylabel("PC{}".format(y))
    plt.legend()
    plt.savefig("result/pca/{}pca{}_{}.png".format(dir_name, x, y))
    #plt.show()
    plt.clf()
