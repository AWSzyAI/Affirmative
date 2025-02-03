import pandas as pd

# 读取CSV文件
path = './data/肯定语体系【Caritas】 - 场景需求表V2.csv'
df = pd.read_csv(path, encoding='utf-8')

# 创建index列
# df['index'] = df['用户问题/症状'] + df['用户1级需求'] + df['用户2级需求']
# 类别,场景,子场景,场景描述,用户需求,心理作用机制与功能,
df['index'] = df['类别'] + df['场景'] + df['子场景'] + df['场景描述'] + df['用户需求'] + df['心理作用机制与功能']
# 按照index列去重
df = df.drop_duplicates(subset='index', keep='first')

# 重置索引（可选）
df = df.reset_index(drop=True)

# 保存处理后的DataFrame到新的CSV文件
df.to_csv(path.replace('.csv','_去重后.csv'), index=False, encoding='utf-8-sig')

print("文件已保存！")