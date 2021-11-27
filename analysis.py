from load_save import load_dataset
from pca import pca_analysis, pca_plot
from tree import lightgbm
from regression import predict_score

import copy
import matplotlib.pyplot as plt


# --spiデータを取り扱うクラス
# --NaNを埋めたりとか、データの操作をする
class SpiData(object):
    def __init__(self, path):
        # --DataFrame型のデータ
        self.df = load_dataset(path)
        self.df_dict = None

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

    # --特定の列でグループ化する
    def divide_df(self, column_name):
        if self.df_dict is None:
            self.df_dict = dict()
            for name, group in self.df.groupby(column_name):
                self.df_dict[name] = copy.deepcopy(group)
                self.df_dict[name] = self.df_dict[name].drop(columns=[column_name])
        else:
            #print(column_name)
            new_dict = dict()
            for key in self.df_dict:
                df = copy.deepcopy(self.df_dict[key])
                for name, group in df.groupby(column_name):
                    new_dict[str(key) + '_' + str(name)] = copy.deepcopy(group)
                    new_dict[str(key) + '_' + str(name)] = \
                        new_dict[str(key) + '_' + str(name)].drop(columns=[column_name])
            self.df_dict = dict()
            self.df_dict = copy.deepcopy(new_dict)


# --データを分析するクラス
# --主成分分析とか、分類器の学習とか
class DataAnalysis(object):
    def __init__(self, df):
        self.df = copy.deepcopy(df)

    # --主成分分析
    def pca(self, x, y, dir_name=""):
        dfs, feature = pca_analysis(self.df, dir_name)
        pca_plot(self.df, feature, x, y, "退職", dir_name)

    # --LightGBM
    def lightgbm(self, ans_column, dir_name):
        lightgbm(self.df, ans_column, dir_name)

    # --column_name別に、退職率を計算
    def count_rate(self, column_name, path):
        plt.rcParams['font.family'] = 'Hiragino sans'
        df_dict = {}
        for name, group in self.df.groupby(column_name):
            df_dict[name] = group

        for column in df_dict:
            x = []
            y = []
            for year, group in df_dict[column].groupby("入行年"):
                x.append(year)
                #cnt = (group["退職"] == 1).sum()
                # y.append(cnt / len(group))
                cnt = group["人事考課"].mean()
                y.append(cnt)
            plt.plot(x, y, label=column)
        plt.legend()
        plt.grid()
        #plt.savefig(path)
        plt.savefig("result/count/性別ごとの人事考課の平均推移")
        plt.show()
        plt.clf()

    # --特定の特徴の年推移
    def transition(self, column_name, path):
        plt.rcParams['font.family'] = 'Hiragino sans'
        if False:
            # --男女別
            for sex, df in self.df.groupby("性別"):
                x = []
                y = []
                for year, group in df.groupby("入行年"):
                    x.append(year)
                    y.append(group[column_name].mean())
                plt.plot(x, y, label=sex)
        else:
            # --男女混合
            x = []
            y = []
            for year, group in self.df.groupby("入行年"):
                x.append(year)
                y.append(group[column_name].mean())
            plt.plot(x, y)
        plt.legend()
        plt.grid()
        plt.xlabel(column_name)
        plt.savefig(path)
        #plt.show()
        plt.clf()

    # --重回帰分析
    def predict_score(self, target_column, dir_name):
        predict_score(self.df, target_column, dir_name)
