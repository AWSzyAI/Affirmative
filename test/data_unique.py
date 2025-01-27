import pandas as pd

# 读取CSV文件
df = pd.read_csv('./data/肯定语体系【Caritas】 - 场景-症状-需求.csv')

# 创建index列
df['index'] = df['用户问题/症状'] + df['用户1级需求'] + df['用户2级需求']

# 按照index列去重
df = df.drop_duplicates(subset='index', keep='first')

# 重置索引（可选）
df = df.reset_index(drop=True)

# 保存处理后的DataFrame到新的CSV文件
df.to_csv('./data/肯定语体系【Caritas】 - 场景-症状-需求_去重后.csv', index=False, encoding='utf-8-sig')

print("文件已保存！")