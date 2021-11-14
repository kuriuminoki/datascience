import copy

import analysis as spi


# --ここでいろいろ分析する
def data_analysis(spi_data):
    # --データ分析クラス作成
    # --考慮する列
    consider_columns = ["Ｗ創造重視"]
    # --考慮しない列
    remove_columns = ["性別", "入行年", "前回", "前々回", "前々々回"]
    data = copy.deepcopy(spi_data)
    # --退職していない人は０
    data.replace_nan(["退職"], 0)
    # --考慮する列、しない列
    data.remove_nan(consider_columns)
    # data.remove_column(remove_columns)
    # --NaNを含む列は全て除去
    data.remove_all_nan()
    print(data.df.shape)
    print(data.df)
    analysis_data = spi.DataAnalysis(data.df)

    # --主成分分析
    analysis_data.pca(1, 2)


# --メイン関数
def main():
    # --データのロード
    path = "data/spiデータ.csv"
    spi_data = spi.SpiData(path)

    # --いろいろ分析する
    data_analysis(spi_data)


if __name__ == '__main__':
    main()
