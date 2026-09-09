import pandas as pd
from sklearn.metrics import cohen_kappa_score

# 満足のラベル（手動で与えられたもの）
satisfied_0_99   = [0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0]
satisfied_100_199 = [0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]

# WRIMEからJoyラベルを取り出す
df = pd.read_csv(
    './wrime-ver2.tsv',
    sep='\t',
    nrows=200,
    usecols=['Writer_Joy']
)

# 二値化（0より大きければ1）
joy_0_99    = (df['Writer_Joy'].iloc[:100] > 0).astype(int).tolist()
joy_100_199 = (df['Writer_Joy'].iloc[100:200] > 0).astype(int).tolist()

# カッパ値を計算
kappa_0_99    = cohen_kappa_score(joy_0_99, satisfied_0_99)
kappa_100_199 = cohen_kappa_score(joy_100_199, satisfied_100_199)

print(f"Cohen's Kappa (0~99行):    {kappa_0_99:.4f}")
print(f"Cohen's Kappa (100~199行): {kappa_100_199:.4f}")

# 参考：各ラベルの1の数
print(f"\n--- 0~99行 ---")
print(f"  Joy の1の数:      {sum(joy_0_99)}")
print(f"  Satisfied の1の数: {sum(satisfied_0_99)}")

print(f"\n--- 100~199行 ---")
print(f"  Joy の1の数:      {sum(joy_100_199)}")
print(f"  Satisfied の1の数: {sum(satisfied_100_199)}")
