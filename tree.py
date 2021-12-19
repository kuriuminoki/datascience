
# #### LightGBM や XGBoost などの決定木による分類 # ####

from load_save import save_data

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import plot_partial_dependence

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


path = "result/tree/lightgbm/"


def lightgbm(df, ans_column, dir_name):
    # --データを用意する
    y = df[ans_column]
    # --訓練データ
    x = df.drop(columns=ans_column)

    # --データをホールドアウト法で分割
    train_x, test_x, train_y, test_y = train_test_split(x, y,  # --訓練データとテストデータに分割する
                                                        test_size=0.3,  # --テストデータの割合
                                                        shuffle=True,  # --シャッフルする
                                                        random_state=21)  # --乱数シードを固定する
    # --データセットを登録
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_test = lgb.Dataset(test_x, test_y, reference=lgb_train)

    # --LightGBMのハイパーパラメータを設定
    # --num_leaves: 葉の数。高すぎると過学習、低すぎると未学習。max_leavesと一緒に調整するべし
    # --min_data_in_leaf: 葉の最小データ数。高いと決定木が深く育つのを抑える
    # --max_depth: 決定機の深さ。他のパラメータとのバランスを考えるべし
    params = {
        # --二値分類問題
        'objective': 'binary',
        # --AUC の最大化を目指す
        'metric': 'auc',
        # --Fatal の場合出力
        'verbosity': -1,
        # --ブースティング
        'boosting_type': 'gbdt',
        # --葉の数
        'num_leaves': 16,
        # --葉の最小データ数
        'min_data_in_leaf': 4,
        # --決定機の深さ
        'max_depth': 16,
    }

    model = lgb.train(params, lgb_train, valid_sets=lgb_test,
                      verbose_eval=50,  # --50イテレーション毎に学習結果出力
                      num_boost_round=1000,  # --最大イテレーション回数指定
                      early_stopping_rounds=100
                      )

    plot_lightgbm(model, dir_name)

    # テストデータを予測する
    pred_y = model.predict(test_x, num_iteration=model.best_iteration)
    #print(pred_y)
    ans_list = list(test_y.round())
    pred_list = list(np.round(pred_y))
    print(ans_list)
    result = metrics.confusion_matrix(ans_list, pred_list)
    print(result)

    # AUC (Area Under the Curve) を計算する
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
    auc = metrics.auc(fpr, tpr)
    #print(auc)

    # ROC曲線をプロット
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig(path + "{}ROC.png".format(dir_name))

    # --特徴量の重要度出力
    plt.figure(figsize=(16, 12))
    importance = pd.DataFrame(model.feature_importance(importance_type="gain"), index=x.columns,
                              columns=['importance']).sort_values(by="importance")
    #print(importance)
    plt.rcParams['font.family'] = 'Hiragino sans'
    plt.barh(importance.index, importance["importance"])
    plt.grid()
    plt.savefig(path + "{}feature.png".format(dir_name))
    #plt.show()
    plt.clf()

    # --PDP
    for c in x.columns:
        plot_pdp(model, x, c, dir_name)


def plot_lightgbm(model, dir_name):
    lgb.plot_tree(model, tree_index=model.best_iteration-1, figsize=(20, 20), show_info=['split_gain'])
    plt.savefig(path + "{}tree.png".format(dir_name))
    #plt.show()
    plt.clf()


def plot_pdp(model, feature, target, dir_name):
    # いくつかの行について、特定のカラムの値を入れかえながらモデルに予測させる
    sampling_factor = 0.5
    resolution = 10
    min_, max_ = feature[target].quantile([0, 1])
    candidate_values = np.linspace(min_, max_, resolution)
    sampled_df = feature.sample(frac=sampling_factor)
    y_preds = np.zeros((len(sampled_df), resolution))
    for index, (_, target_row) in enumerate(sampled_df.iterrows()):
        for trial, candidate_value in enumerate(candidate_values):
            target_row[target] = candidate_value
            y_preds[index][trial] = model.predict([target_row])

    # 予測させた結果をプロットする
    mean_y_preds = y_preds.mean(axis=0)  # 平均
    sd_y_preds = y_preds.std(axis=0)  # 標準偏差
    # 平均 ± 1 SD を折れ線グラフにする
    plt.plot(candidate_values, mean_y_preds)
    plt.fill_between(candidate_values,
                     mean_y_preds - sd_y_preds,
                     mean_y_preds + sd_y_preds,
                     alpha=0.5)
    plt.xlabel('factor')
    plt.ylabel('target')
    plt.savefig(path + "{}pdp_{}.png".format(dir_name, target))
    #plt.show()
    plt.clf()
