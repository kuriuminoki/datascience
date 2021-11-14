import pandas as pd


# CSVファイルのロード DataFrame型で取得
def load_dataset(filename):
    df = pd.read_csv(filename, encoding='shift-jis')
    return df


# --ファイルの保存
def save_data(df, path):
    df.to_csv(path, encoding="shift_jis")
