#wrime-ver2.tsvの101〜200行目を対象に、原田：１、南：０のインデックスを見つける、対応するセンテンスの表示

import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score

M = [0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0]
H = [0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0]

target_columns = ['Sentence']

df = pd.read_csv(
    './wrime-ver2.tsv',
    sep='\t',
    skiprows=range(1, 101),  
    nrows=100,
    usecols=target_columns
)

tp_indices = [i for i, (m, h) in enumerate(zip(M, H)) if m == 0 and h == 1]
print(f"TP indices (0-based): {tp_indices}")
print()

for index, row in df.iterrows():
    sentence = row['Sentence']
    print(f'{index + 100}:{sentence}')