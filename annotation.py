import pandas as pd

target_columns = [
    'idx',
    'is_satisfy'
]

df = pd.read_csv(
    'count_satisfy.csv',
    usecols=target_columns
)

satisfy_cnt = 0

for index, row in df.iterrows():
    if row['is_satisfy'] > 0:
        satisfy_cnt += 1

satisfy_row = df[df['is_satisfy'] > 0]
print(f"satisfy_cnt: {satisfy_cnt}")
print(satisfy_row)
