
# #### LightGBM や XGBoost などの決定木による分類 # ####

from load_save import save_data

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
    params = {
        # --二値分類問題
        'objective': 'binary',
        # --AUC の最大化を目指す
        'metric': 'auc',
        # --Fatal の場合出力
        'verbosity': -1,
    }

    lgb_results = {}  # --学習の履歴を入れる入物
    model = lgb.train(params, lgb_train, valid_sets=lgb_test,
                      verbose_eval=50,  # --50イテレーション毎に学習結果出力
                      num_boost_round=1000,  # --最大イテレーション回数指定
                      early_stopping_rounds=100
                      )

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


