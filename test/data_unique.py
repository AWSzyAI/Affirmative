# import pandas as pd

# # 读取CSV文件
# path = './data/肯定语体系【Caritas】 - 场景需求表V2.csv'
# df = pd.read_csv(path, encoding='utf-8')
# print(df.shape)
# # 创建index列
# # df['index'] = df['用户问题/症状'] + df['用户1级需求'] + df['用户2级需求']
# df['index'] = df['类别'] + df['场景'] + df['子场景'] + df['场景描述'] + df['用户需求'] + df['心理作用机制与功能'] + df['句子级别']
# # 按照index列去重
# df = df.drop_duplicates(subset='index', keep='first')

# # 重置索引（可选）
# df = df.reset_index(drop=True)

# # 保存处理后的DataFrame到新的CSV文件
# df.to_csv(path.replace('.csv','_去重后.csv'), index=False, encoding='utf-8-sig')

# print("文件已保存！")

import pandas as pd

# 读取CSV文件
path = './data/肯定语体系【Caritas】 - 场景需求表V2.csv'
df = pd.read_csv(path, encoding='utf-8')

# 填充空值为'NA'，避免空字符串导致误判重复
df.fillna('NA', inplace=True)

# 创建index列（根据实际业务调整拼接列）

df['index'] = df['类别'] + df['场景'] + df['子场景'] + df['场景描述'] + df['用户需求'] + df['心理作用机制与功能'] + str(df['句子级别'])

# 或者使用部分有区分度的列（需根据数据实际情况选择）
# df['index'] = df['类别'] + df['场景'] + df['用户需求']

# 按照index列去重
df = df.drop_duplicates(subset='index', keep='first')
df = df[['类别', '场景', '子场景', '场景描述', '用户需求', '心理作用机制与功能', '句子级别']]
# 重置索引（可选）
df = df.reset_index(drop=True)

# 保存处理后的DataFrame到新的CSV文件
df.to_csv(path.replace('.csv','_去重后.csv'), index=False, encoding='utf-8-sig')

print("文件已保存！去重后行数:", len(df))