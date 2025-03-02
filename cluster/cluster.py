import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
from datetime import datetime
import coloredlogs
import sys
# 设置路径
sys.path.append('../src')
from utils import make_Affirmative_by_need, make_Affirmative, matched_paradigms
from prompt import get_role_prompt




# 配置日志记录
log_file = './Log/' + datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 创建 logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 创建控制台输出处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到 logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 为了彩色日志输出
coloredlogs.install(level='INFO', logger=logger)

# 载入数据
with open('../data/fav_article_20.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
articles = data['article']

# 设置匹配的范式
# matched_paradigms = [
#     '情绪应对式: 简单-情绪应对式',
#     "情绪应对式: 认知行为-情绪应对式",
#     "情绪应对式: 正念-情绪应对式",
#     "情绪应对式: 道家-情绪应对式",
#     '安抚接纳式: 简单-自我接纳式',
#     '安抚接纳式: 简单-环境接纳式',
#     "外源锚定式: 简单-外源锚定式",
#     "心态稳定式: 简单-心态稳定式",
#     '积极感知式: 简单-积极感知式',
#     '主体自信式: 简单-主体自信式',
#     '主体自信式: 逻辑-主体自信式',
#     '潜能确认式: 简单-潜能确认式',
#     "心念成长式: 简单-心念培育式",
#     "心念成长式: 逻辑-心念培育式",
#     "心念成长式: 简单-心念锚定式",
#     "心念成长式: 逻辑-心念锚定式",
#     "行动宣告式: 简单-行动宣告式",
#     "自然改变式: 简单-自然改变式",
#     "意义构建式: 简单-主体意义式",
#     "意义构建式: 简单-经历意义式",
#     "价值锚定式: 简单-价值锚定式",
#     "价值锚定式: 逻辑-价值锚定式",
#     "感恩整合式: 简单-感恩整合式",
#     "独特价值宣言式: 简单-独特价值宣言式",
#     "主权宣告式: 简单-主权宣告式",
#     "对抗超升式: 权力意志-对抗超升式",
#     "爱之循环式: 简单-爱之循环式",
# ]

# 设置文件路径
csv_file = "../data/fav_article_20.csv"
paradigm_md_path = "../data/paradigm.md"

# 并发执行每篇文章的处理任务
def process_article(article):
    title = article.get("title")
    logger.info(f"开始处理文章: {title}")
    zhihu_link = article.get("zhihu_link")
    content = article.get("content")

    role_maker = "Affirmative_maker-0213"
    messages = []
    message = f" ##【必须满足的要求】基于下面参考文章的原句进行修改二创，最终结果以单句为主，不要超过两个分句。不得和范式中的例句有重复的内容。 ## 文章 标题：{title} \n 正文：{content}"

    role_prompt = get_role_prompt(role_maker, articles=articles)

    # 如果messages为空（None 或 空列表[]），则初始化
    if not messages:
        messages = [{"role": "system", "content": role_prompt}, {"role": "user", "content": message}]
    else:
        messages.append({"role": "user", "content": message})

    # 并发处理每个范式
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for paradigm in matched_paradigms:
            # 构造症状字典
            symptom = {
                "场景": '',
                "子场景": '',
                '用户需求': '',
                '场景描述': '',
                '心理作用机制与功能': '',
                '句子级别': '',
                '句子范式': ''
            }
            # 使用 executor 异步处理每个范式
            future = executor.submit(make_Affirmative_by_need, 
                                     symptom, 
                                     paradigm, 
                                     sentences=None, 
                                     zhihu_link=zhihu_link, 
                                     output_file=csv_file, 
                                     paradigm_md_path=paradigm_md_path,
                                     messages=messages)
            futures.append(future)

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取任务的返回结果
                logger.info(f"成功处理范式: {paradigm}")
            except Exception as e:
                logger.error(f"处理范式 {paradigm} 时发生错误: {e}")

# 主程序：并发处理每篇文章
if __name__ == "__main__":
    logger.info("程序开始运行")
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_article, article) for article in articles]
        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取每个任务的返回结果
            except Exception as e:
                logger.error(f"处理文章时发生错误: {e}")
    logger.info("程序运行结束")