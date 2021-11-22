
# #### LightGBM や XGBoost などの決定木による分類 # ####

from load_save import save_data

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from matplotlib import pyplot as plt


def lightgbm(df, ans_column):
    # --データを用意する
    y = df[ans_column]
    # --訓練データ
    x = df.drop(columns=ans_column)

    # --データをホールドアウト法で分割
    train_x, test_x, train_y, test_y = train_test_split(x, y,  # --訓練データとテストデータに分割する
                                                        test_size=0.3,  # --テストデータの割合
                                                        shuffle=True,  # --シャッフルする
                                                        random_state=0)  # --乱数シードを固定する
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
        'max_depth': 8,
    }

    model = lgb.train(params, lgb_train, valid_sets=lgb_test,
                      verbose_eval=50,  # --50イテレーション毎に学習結果出力
                      num_boost_round=1000,  # --最大イテレーション回数指定
                      early_stopping_rounds=100
                      )

    plot_lightgbm(model, train_x)

    # テストデータを予測する
    pred_y = model.predict(test_x, num_iteration=model.best_iteration)

    # AUC (Area Under the Curve) を計算する
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    # ROC曲線をプロット
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig("result/tree/lightgbm/ROC.png")

    # --特徴量の重要度出力
    print(model.feature_importance())
    # --特徴量の重要度をプロット
    plt.rcParams['font.family'] = 'Hiragino sans'
    lgb.plot_importance(model)
    plt.savefig("result/tree/lightgbm/feature.png")
    plt.show()


def plot_lightgbm(model, x):
    #export_graphviz(model, out_file="tree.dot", feature_names=x.columns, class_names=["0", "1"], filled=True,
     #               rounded=True)

    #graph = pydotplus.graph_from_dot_file(path="tree.dot")
    #Image(graph.create_png())
    ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 20), show_info=['split_gain'])
    plt.show()
    graph = lgb.create_tree_digraph(model, tree_index=0, format='png', name='Tree')
    graph.render(view=True)
