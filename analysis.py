from load_save import load_dataset
from pca import pca_analysis, pca_plot

import copy


# --spiデータを取り扱うクラス
# --NaNを埋めたりとか、データの操作をする
class SpiData(object):
    def __init__(self, path):
        # --DataFrame型のデータ
        self.df = load_dataset(path)

    # --一つでもNaNがある列を削除
    def remove_all_nan(self):
        self.df = self.df.dropna(how='any', axis=1)

    # --column_namesの列にNaNがある行を除去
    def remove_nan(self, column_names):
        self.df = self.df.dropna(subset=column_names)

    # --NaNをすべてvalueに置換
    def replace_all_nan(self, value=0):
        self.df = self.df.fillna(value)

    # --特定の列のNaNをすべてvalueに置換
    def replace_nan(self, column_name, value=0):
        for name in column_name:
            self.df = self.df.fillna({name: value})

    # --特定の列を削除
    def remove_column(self, column_names):
        self.df = self.df.drop(columns=column_names)

    # --一部の列をone-hot表現に変更
    def add_one_hot(self):
        # --入行年をone-hot表現にする
        for year in range(2007, 2020):
            self.df[str(year)] = 0
            for i in range(len(self.df)):
                if self.df['入行年'][i] == year:
                    self.df[str(year)][i] = 1
        self.remove_column('入行年')
        # --性別をone-hot表現にする
        self.df['男'] = 0
        self.df['女'] = 0
        for i in range(len(self.df)):
            if self.df['性別'][i] == 1:
                self.df['男'][i] = 1
            else:
                self.df['女'][i] = 1
        self.remove_column('性別')


# --データを分析するクラス
# --主成分分析とか、分類器の学習とか
class DataAnalysis(object):
    def __init__(self, df):
        self.df = copy.deepcopy(df)

    def pca(self, x, y):
        dfs, feature = pca_analysis(self.df)
        pca_plot(self.df, feature, x, y, "退職")
