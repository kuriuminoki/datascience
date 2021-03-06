from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# ##### 重回帰分析による予測 ########## #


path = "result/regression/伸びしろ/"


# --正規化
def normalize(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


# --重回帰分析
def predict_score(df, target_column, dir_name):
    df_np = normalize(df)
    x_np = df_np.drop(target_column, axis=1)
    y_np = df_np[target_column]
    model = LinearRegression()
    model.fit(x_np, y_np)
    result = pd.DataFrame({"Name": x_np.columns, "Coefficients": model.coef_})\
        .sort_values(by="Coefficients")
    # print(result)
    # print(model.intercept_)
    score = model.score(x_np, y_np)
    # print(score)
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = 'Hiragino sans'
    plt.barh(result["Name"], result["Coefficients"])
    plt.grid()
    plt.savefig(path + "{}{}_係数.png".format(dir_name, target_column))
    #plt.show()
    plt.clf()

    # --実際に予測するとどうなったか
    y_pred = model.predict(x_np)
    y_pred = list(y_pred)
    y_np = list(y_np)
    plt.scatter(y_np, y_np, c='red')
    plt.scatter(y_np, y_pred, c='blue', alpha=0.5)
    plt.grid()
    plt.savefig(path + "{}{}_plot.png".format(dir_name, target_column))
    plt.clf()
    # --評価
    for feature in x_np.columns:
        plt.scatter(df[feature], y_np, marker='+', color="b", label="データ")
        plt.scatter(df[feature], y_pred, color="r", label="予測値")
        plt.legend()
        plt.grid()
        plt.savefig(path + "{}{}_{}と予測結果.png".format(dir_name, target_column, feature))
        # plt.show()
        plt.clf()

