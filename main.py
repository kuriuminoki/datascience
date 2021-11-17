import copy

import analysis as spi


# --一部の特徴をまとめたもの
category1 = ['社会的内向性', '社会的内向性', '身体活動性', '持続性', '慎重性',
             '達成意欲', '活動意欲', '敏感性', '自責性', '気分性',
             '独自性', '自信性', '高揚性']

category2 = ['内向-外向', '大胆-慎重', '静的-動的', '淡白-持続', '現実-理想',
             '強靭-繊細', '他者-自己', '安定-高揚']

category3 = ['言語(得点)', '言語(段階)', '非言語(得点)', '非言語(段階)', '総合(得点)']

category4 = ['総合(段階)', '職適(Ａ)', '職適(Ｂ)', '職適(Ｃ)', '職適(Ｄ)',
             '職適(Ｅ)', '職適(Ｆ)', '職適(Ｇ)', '職適(Ｈ)', '職適(Ｉ)',
             '職適(Ｊ)', '職適(Ｋ)', '職適(Ｌ)', '職適(Ｍ)', '職適(Ｎ)']

category5 = ['Ｗ創造重視', 'Ｘ結果重視', 'Ｙ調和重視', 'Ｚ秩序重視']

category6 = ['従順性', '回避性', '批判性', '自己尊重性', '懐疑思考性']

category7 = ['親和-独立', '気長-機敏', '柔軟-持続', '現実-挑戦', '自信-感受']

category8 = ['人事考課', '前回', '前々回', '前々々回', '平均']


# --ここでいろいろ分析する
def data_analysis(spi_data):
    # --データ分析クラス作成

    # ########主成分分析############ #
    # --考慮する列
    consider_columns = ["Ｗ創造重視"]
    # --考慮しない列
    remove_columns = ["性別", "入行年", "前回", "前々回", "前々々回"]
    # --データ取得
    data = copy.deepcopy(spi_data)
    # --退職していない人は０
    data.replace_nan(["退職"], 0)
    # --考慮する列、しない列
    data.remove_nan(consider_columns)
    # data.remove_column(remove_columns)
    # --NaNを含む列は全て除去
    data.remove_all_nan()
    # --主成分分析を実行
    analysis_data = spi.DataAnalysis(data.df)
    #analysis_data.pca(1, 2)
    #analysis_data.pca(1, 8)
    #analysis_data.pca(1, 9)
    #analysis_data.pca(1, 10)
    #analysis_data.pca(1, 11)

    # #######機械学習########## #
    data = copy.deepcopy(spi_data)
    # --退職していない人は０
    data.replace_nan(["退職"], 0)
    # --年と性別をone-hot表現にする
    # data.add_one_hot()
    # --NaNを含む列は全て除去
    data.remove_all_nan()
    # --LightGBM
    analysis_data = spi.DataAnalysis(data.df)
    analysis_data.lightgbm("退職")


# --メイン関数
def main():
    # --データのロード
    path = "data/spiデータ.csv"
    spi_data = spi.SpiData(path)

    # --いろいろ分析する
    data_analysis(spi_data)


if __name__ == '__main__':
    main()
