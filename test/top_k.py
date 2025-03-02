import json
import pandas as pd

n = 20

# 读取原始数据文件
with open('article_202502250955.json', 'r', encoding='utf-8') as f:
    articles = json.load(f)['article']  # 根据数据结构获取文章列表

# 处理收藏数为None的情况，转换为0
for article in articles:
    if article['favorites_count'] is None:
        article['favorites_count'] = 0
    if article['read_count'] is None:
        article['read_count'] = 0


# type = "read_count"
type = "favorites_count"

# 按照阅读数/收藏数降序排序
sorted_articles = sorted(articles, 
                        key=lambda x: x[type], 
                        reverse=True)

# 取前n条数据
top_n = sorted_articles[:n]

# 保存结果
with open(f'data_{n}.json', 'w', encoding='utf-8') as f:
    json.dump({'article': top_n}, f, 
             ensure_ascii=False, 
             indent=4)
# 保存"read_count"，"favorites_count"，"title"，"question"，"zhihu_link"到csv
df = pd.DataFrame(top_n)
df = df[["read_count", "favorites_count", "title", "question", "zhihu_link"]]
df.to_csv(f"data_{n}.csv", index=False, encoding='utf-8-sig')
print("处理完成，共筛选出{}条数据".format(len(top_n)))



