import pandas as pd

df = pd.read_csv('../data/标注结果_batch_1.csv')
for i in range(59):
    df_tmp = pd.read_csv(f'标注结果_batch_{2+i}.csv')
    df.extend(df_tmp)

df.to_csv('../data/merged_5800.csv')