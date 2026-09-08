import pandas as pd

M = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
H = [0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0]

target_columns = ['Sentence']

df = pd.read_csv(
    './wrime-ver2.tsv',
    sep='\t',
    skiprows=range(1, 101),  # ヘッダーは残してデータの1〜100行目をスキップ
    nrows=100,
    usecols=target_columns
)

tp_indices = [i for i, (m, h) in enumerate(zip(M, H)) if m == 1 and h == 1]
print(f"TP indices (0-based): {tp_indices}")
print()

for index, row in df.iterrows():
    sentence = row['Sentence']
    print(f'{index + 100}:{sentence}')
