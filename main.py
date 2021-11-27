import copy

import analysis as spi

import os


# --一部の特徴をまとめたもの
category0 = ['通番', '退職', '入行年', '性別']

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

category9 = ['前回', '前々回', '前々々回', '平均']


# ###### データのカウント ####### #
def count_rate(spi_data):
    # --データ取得
    data = copy.deepcopy(spi_data)
    data.replace_nan(["退職"], 0)
    analysis_data = spi.DataAnalysis(data.df)
    analysis_data.count_rate("性別", "result/count/性別ごとの退職率推移.png")


# ###### 各特徴の平均の推移 ##### #
def transition(spi_data):
    # --データ取得
    data = copy.deepcopy(spi_data)
    data.replace_nan(["退職"], 0)
    analysis_data = spi.DataAnalysis(data.df)
    for column in data.df.columns:
        # analysis_data.transition(column, "result/transition/男女別/{}.png".format(column))
        analysis_data.transition(column, "result/transition/男女混合/{}.png".format(column))


# ####### 主成分分析 ######### #
def pca(spi_data):
    # --データ取得
    data = copy.deepcopy(spi_data)
    # --退職していない人は０
    data.replace_nan(["退職"], 0)
    # --考慮する列
    consider_columns = ["Ｗ創造重視"]
    # data.remove_nan(consider_columns)
    # --考慮しない列
    remove_columns = ["性別", "入行年", "前回", "前々回", "前々々回"]
    # data.remove_column(remove_columns)
    # --NaNを含む列は全て除去
    data.remove_all_nan()
    # --主成分分析を実行
    if True:
        # --何かでグループ化して個別に分析するとき
        divide_column = ["入行年", "性別"]
        for c in divide_column:
            data.divide_df(c)
        data_dict = copy.deepcopy(data.df_dict)
        for name in data_dict:
            print(name)
            analysis_data = spi.DataAnalysis(data_dict[name])
            #os.mkdir("result/pca/{}".format(str(name) + '_'.join(divide_column)))
            analysis_data.pca(3, 4, str(name) + '_'.join(divide_column) + '/')
    else:
        # --グループ化しないとき
        analysis_data = spi.DataAnalysis(data.df)
        analysis_data.pca(1, 2, "総合/")


# ######## LightGBM ########## #
def lightgbm(spi_data):
    # --データ取得
    data = copy.deepcopy(spi_data)
    # --退職していない人は０
    data.replace_nan(["退職"], 0)
    # --考慮する列
    consider_columns = ["Ｗ創造重視"]
    # data.remove_nan(consider_columns)
    # --考慮しない列
    remove_columns = ["通番"] + category8
    data.remove_column(remove_columns)
    # --年と性別をone-hot表現にする
    # data.add_one_hot()
    # --LightGBM
    target = "退職"
    if True:
        # --何かでグループ化して個別に分析するとき
        divide_column = ["入行年", "性別"]
        for c in divide_column:
            data.divide_df(c)
        data_dict = copy.deepcopy(data.df_dict)
        for name in data_dict:
            print(name)
            analysis_data = spi.DataAnalysis(data_dict[name])
            #os.mkdir("result/tree/lightgbm/{}".format(str(name) + '_'.join(divide_column)))
            analysis_data.lightgbm(target, str(name) + '_'.join(divide_column) + '/')
    else:
        # --グループ化しないとき
        analysis_data = spi.DataAnalysis(data.df)
        analysis_data.lightgbm(target, "総合/")


# ######## 重回帰分析 ############# #
def predict_score(spi_data):
    # --データ取得
    data = copy.deepcopy(spi_data)
    # --退職していない人は０
    # data.replace_nan(["退職"], 0)
    # --考慮する列
    consider_columns = ["Ｗ創造重視"]
    # data.remove_nan(consider_columns)
    # --考慮しない列
    remove_columns = ["通番"] + category9
    data.remove_column(remove_columns)
    # --重回帰分析
    data.remove_nan(["人事考課"])
    target = "人事考課"
    if True:
        # --何かでグループ化して個別に分析するとき
        divide_column = ["性別"]
        for c in divide_column:
            data.divide_df(c)
        data_dict = copy.deepcopy(data.df_dict)
        for name in data_dict:
            print(name)
            data_dict[name] = data_dict[name].dropna(how='any', axis=1)
            analysis_data = spi.DataAnalysis(data_dict[name])
            #os.mkdir("result/regression/{}".format(str(name) + '_'.join(divide_column)))
            analysis_data.predict_score(target, str(name) + '_'.join(divide_column) + '/')
    else:
        # --グループ化しないとき
        # --NaNを含む列は全て除去
        data.remove_all_nan()
        analysis_data = spi.DataAnalysis(data.df)
        analysis_data.predict_score(target, "総合/")


# --ここでいろいろ分析する やらないやつはコメントアウト
def data_analysis(spi_data):
    print("Analysis started.")
    # ########カウント############# #
    #count_rate(spi_data)

    # ########推移############## #
    #transition(spi_data)

    # ########主成分分析############ #
    #pca(spi_data)

    # #######機械学習########## #
    #lightgbm(spi_data)

    # #######重回帰分析####### #
    predict_score(spi_data)


# --メイン関数
def main():
    # --データのロード
    path = "data/spiデータ.csv"
    spi_data = spi.SpiData(path)

    # --いろいろ分析する
    data_analysis(spi_data)


if __name__ == '__main__':
    main()
