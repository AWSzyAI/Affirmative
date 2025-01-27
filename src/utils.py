
import os
import csv
import time
import json
import requests
import concurrent
import threading
from tqdm import tqdm
from openai import RateLimitError  # 导入 RateLimitError 异常
from src.milvus_utils import embeddings, query_article_data
from src.kimi_api import client
from src.prompt import get_role_prompt

# DEBUG = True
DEBUG = False

HEADERS = ['自我肯定语', '生产者', '参考需求','用户问题/症状', '用户1级需求', '用户2级需求', 'zhihu_link']
# HEADERS = ['自我肯定语']
max_retries = 3
retry_delay = 5  # 重试延时（秒）
checkpoint_lock = threading.Lock()  # 线程锁，用于保护检查点文件的更新

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# FILE I/O
def make_data_item(user_problem, need_1, need_2, need, self_affirmative_phrase, type, zhihu_link=None, think_log=None):
    """构造数据项"""
    if type=='3':
        return {
            '用户问题/症状': user_problem,
            '用户1级需求': need_1,
            '用户2级需求': need_2,
            '参考需求': need,
            '自我肯定语': self_affirmative_phrase,
            '生产者':'3号',
            'zhihu_link': zhihu_link,
            '反思日志': think_log
        }
    elif type=='2':
        return {
            '用户问题/症状': user_problem,
            '用户1级需求': need_1,
            '用户2级需求': need_2,
            '参考需求': need,
            '自我肯定语': self_affirmative_phrase,
            '生产者':'2号',
            'zhihu_link': zhihu_link,
            '反思日志': think_log
        }

def load_csv(file_path):
    """加载CSV文件，返回列表格式的数据"""
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def save_to_csv(output_file, data_item,HEADERS):
    """将生成的自我肯定语及其对应数据保存到CSV文件"""
    # print(f"save_to_csv({output_file}, data_item,HEADERS):")
    with open(output_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # 如果文件为空，则写入表头
        if file.tell() == 0:
            writer.writerow(HEADERS)

        cleaned_row = [clean_value(data_item.get(header, '')) for header in HEADERS]
        writer.writerow(cleaned_row)

# Warpper
def clean_value(value):
    """清理字段中的换行符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ')  # 删除换行符和回车符
    return value

def remove_duplicates(article_data):
    """
    根据文章的 id 去重
    :param article_data: 包含文章数据的列表
    :return: 去重后的文章数据列表
    """
    seen_ids = set()  # 用于存储已经见过的 id
    unique_articles = []  # 用于存储去重后的文章

    for article in article_data:
        article_id = article['entity']['id']  # 获取文章的 id
        if article_id not in seen_ids:
            seen_ids.add(article_id)  # 将 id 添加到已见集合
            unique_articles.append(article)  # 添加到去重后的列表

    return unique_articles

# 断点续传
def get_checkpoint(checkpoint_file):
    """获取 checkpoint 文件中记录的所有已完成索引"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data
    return []

def update_checkpoint(checkpoint_file, index):
    """更新 checkpoint 文件（线程安全）"""
    with checkpoint_lock:  # 使用线程锁确保线程安全
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
        else:
            checkpoint_data = []
        
        if index not in checkpoint_data:
            checkpoint_data.append(index)

        # 写回检查点文件
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

# Caritas API
def get_encouragements(message, k=5):
    """利用 message 检索鼓励语 quote"""
    encouragement_url = "http://test.caritas.pro:5001/query_excerpt_data"
    payload = {
        "query_text": message,
        "top_k": k
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(encouragement_url, json=payload, headers=headers)
    encouragement_quotes = []
    if response.status_code == 200:
        data = response.json()
        for item in data["data"]:
            encouragement_quotes.append(item["entity"]["quote"])
    else:
        print("请求失败: %s %s", response.status_code, response.text)
    return encouragement_quotes

def query_article(query_text, top_k=2):
    """查询文章数据"""
    try:
        query_vector = embeddings.embed_query(query_text)
        article_data = query_article_data('article_collection', query_vector, top_k)
        # debug(article_data)

        # 过滤掉包含 "想法集" 标签的文章
        filtered_article_data = [
            article for article in article_data 
            if "想法集" not in article['entity']['tags']
        ]
        if not filtered_article_data:
            print('未检索到articles')
            return []
        if len(filtered_article_data) < len(article_data):
            print("Found '想法集' tag, removed from results.")
        return filtered_article_data
    except Exception as e:
        print("Error: %s", e)
        return []

# main Job
# def generate_self_affirmative_phrase_concurrent(symptoms_file, csv_file, checkpoint_file, n, delay, max_retries, DEBUG):
#     symptoms_data = load_csv(symptoms_file)

#     completed_indices = set(get_checkpoint(checkpoint_file)) 
#     print(f"从检查点文件读取到已完成的索引: {completed_indices}")
    
#     with tqdm(total=len(symptoms_data), initial=len(completed_indices), desc="生成进度", unit="症状", position=0) as pbar:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = {}
#             for i in range(len(symptoms_data)):
#                 if i in completed_indices:  # 如果任务已完成，跳过
#                     pbar.update(1)  # 更新进度条
#                     continue
#                 symptom = symptoms_data[i]

#                 user_problem = symptom['用户问题/症状']
#                 need_1 = symptom['用户1级需求']
#                 need_2 = symptom['用户2级需求']

#                 future = executor.submit(
#                     generate_affirmation_for_symptom, i, user_problem, n, delay, max_retries, csv_file, checkpoint_file, need_1, need_2, DEBUG=DEBUG
#                 )
#                 futures[future] = i  # 将 future 和索引关联起来

#             for future in concurrent.futures.as_completed(futures):
#                 index = futures[future]  # 获取当前任务的索引
#                 try:
#                     future.result()  # 捕获异常，如果任务有异常，会抛出
#                 except Exception as e:
#                     print(f"任务 {index} 失败: {e}")
#                 finally:
#                     # with checkpoint_lock:
#                     pbar.update(1)  # 更新进度条
#                     update_checkpoint(checkpoint_file, index)  # 更新检查点文件

#     print(f"所有未生成过的自我肯定语已保存到 {csv_file}")
#     if os.path.exists(checkpoint_file):
#         os.remove(checkpoint_file)
#         print(f"已删除文件: {checkpoint_file}")
#     else:
#         print(f"文件不存在: {checkpoint_file}")

def generate_self_affirmative_phrase_concurrent(symptoms_file, csv_file, checkpoint_file, n, delay, max_retries, DEBUG, use_concurrency=False):
    symptoms_data = load_csv(symptoms_file)

    completed_indices = set(get_checkpoint(checkpoint_file)) 
    print(f"从检查点文件读取到已完成的索引: {completed_indices}")
    
    with tqdm(total=len(symptoms_data), initial=len(completed_indices), desc="生成进度", unit="症状", position=0) as pbar:
        if use_concurrency:
            # 并发执行逻辑
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                for i in range(len(symptoms_data)):
                    if i in completed_indices:  # 如果任务已完成，跳过
                        pbar.update(1)  # 更新进度条
                        continue
                    symptom = symptoms_data[i]

                    user_problem = symptom['用户问题/症状']
                    need_1 = symptom['用户1级需求']
                    need_2 = symptom['用户2级需求']

                    future = executor.submit(
                        generate_affirmation_for_symptom, i, user_problem, n, delay, max_retries, csv_file, checkpoint_file, need_1, need_2, DEBUG=DEBUG
                    )
                    futures[future] = i  # 将 future 和索引关联起来

                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]  # 获取当前任务的索引
                    try:
                        future.result()  # 捕获异常，如果任务有异常，会抛出
                    except Exception as e:
                        print(f"任务 {index} 失败: {e}")
                    finally:
                        pbar.update(1)  # 更新进度条
                        update_checkpoint(checkpoint_file, index)  # 更新检查点文件
        else:
            # 串行执行逻辑
            for i in range(len(symptoms_data)):
                if i in completed_indices:  # 如果任务已完成，跳过
                    pbar.update(1)  # 更新进度条
                    continue
                symptom = symptoms_data[i]

                user_problem = symptom['用户问题/症状']
                need_1 = symptom['用户1级需求']
                need_2 = symptom['用户2级需求']

                try:
                    generate_affirmation_for_symptom(i, user_problem, n, delay, max_retries, csv_file, checkpoint_file, need_1, need_2, DEBUG=DEBUG)
                except Exception as e:
                    print(f"任务 {i} 失败: {e}")
                finally:
                    pbar.update(1)  # 更新进度条
                    update_checkpoint(checkpoint_file, i)  # 更新检查点文件

    print(f"所有未生成过的自我肯定语已保存到 {csv_file}")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"已删除文件: {checkpoint_file}")
    else:
        print(f"文件不存在: {checkpoint_file}")

def generate_affirmation_for_symptom(i, user_problem, n, delay, max_retries, csv_file, checkpoint_file, need_1, need_2, DEBUG=False):
    """生成单个症状的自我肯定语并保存到 CSV"""

    think_log = ''
    
    message = '场景/症状：' + user_problem + '1级需求（安慰效果）：' + need_1 + ' 2级需求（鼓励效果）：' + need_2
    article_data = query_article(user_problem, 1)
    print(len(article_data))
    article_data += query_article(need_1, 1)
    article_data += query_article(need_2, 1)
    print(len(article_data))
    unique_article_data = remove_duplicates(article_data)
    print(f"去重后文章数量: {len(unique_article_data)}")
    zhihu_link = ' '.join([article['entity']['zhihu_link'] for article in article_data]) if article_data else "无链接"
    articles = ' '.join([article['entity']['content'] for article in article_data])
    structured_articles = get_structured_articles(article_data, client,"article-structurer")
    
    # debug(structured_article)
    sentences = []
    for i, structured_article in enumerate(structured_articles):
        # debug('发问：思考、反省: %s', structured_article.get('发问：思考、反省', 'N/A'))
        # debug('价值观: %s', structured_article.get('价值观', 'N/A'))
        # debug('行动：可效仿的行动指南: %s', structured_article.get('行动：可效仿的行动指南', 'N/A'))
        # debug('慈悲：理解、接受、宽恕: %s', structured_article.get('慈悲：理解、接受、宽恕', 'N/A'))
        # debug('状态描述：成为这样的我: %s', structured_article.get('状态描述：成为这样的我', 'N/A'))
        
        for j in ['状态描述：成为这样的我']:
            if structured_article.get(j):
                affirmative,messages = make_Affirmative(structured_article.get(j), client,"Affirmative_maker",articles=articles)
                sentences.extend(affirmative)
            else:
                print(f"{j} not found")
    
    # print("sentences:", sentences)
    
    # think_log += clean_value(str(message))
    # think_log += clean_value(str(article_data))
    # think_log += clean_value(str(structured_articles))
    # think_log += clean_value(str(sentences))
    
    
    need = "1级需求: " + need_1
    think_log_1 = think_log + clean_value(str(need))
    make_Affirmative_by_need(articles, need_1, need_2, need, sentences, client, user_problem, zhihu_link, think_log_1, csv_file,"style-fliter",messages)
    
    need = "2级需求: " + need_2
    think_log_2 = think_log + clean_value(str(need))
    make_Affirmative_by_need(articles, need_1, need_2, need, sentences, client, user_problem, zhihu_link, think_log_2, csv_file,"style-fliter",messages)

#  pipeline LLM API
def get_structured_articles(article_data, client, role):
    max_retries = 3
    retry_delay = 5 
    role_prompt = get_role_prompt(role)
    
    structured_articles = []
    for article in article_data:
        content = article['entity']['content']
        structured_article = ''
        # debug(content) # ok，正常返回文章内容
        for attempt in range(max_retries):
            try:
                # 调用 API 进行结构化转换
                completion = client.chat.completions.create(
                    model="moonshot-v1-auto",
                    messages=[
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": content}
                    ],
                    temperature=1,
                    response_format={"type": "json_object"},  # 确保返回 JSON 格式
                    n=1  # 请求返回1个结果
                )

                # 解析 API 返回的内容
                response = completion.choices[0].message.content.strip()
                # debug(response) # ok
                
                try:
                    structured_article = json.loads(response)  # 将字符串解析为 JSON 对象
                    structured_articles.append(structured_article)
                    # print("279: ", structured_articles)
                    break  # 成功时退出重试循环
                except json.JSONDecodeError:
                    print(f"Error: Failed to parse JSON response for article: {content}")
                    break  # 解析失败时退出重试循环

            except RateLimitError:
                if attempt < max_retries - 1:
                    print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Failed to generate structured article.")
                    raise

        # 添加延时，降低请求速率
        time.sleep(1)  # 每次请求后延时 1 秒

    return structured_articles

def make_Affirmative(content, client,role,articles):
    if not content:
        return [" "]

    role_prompt = get_role_prompt(role,articles=articles)
    message = "句子："+"\n".join(content)+f"\n 文章”“”{articles}“”“，生成10个句子"
    attempt = 0
    while attempt < max_retries:
        try:
            messages=[
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": message}
                ]
            completion = client.chat.completions.create(
                model="moonshot-v1-auto",
                messages = messages,
                temperature=1,
                response_format={"type": "json_object"},  # 确保返回 JSON 格式
                n=1  # 请求返回1个结果
            )
            response = completion.choices[0].message.content.strip()
            # debug("API Response: %s", response)
            try:
                response_dict = json.loads(response)
                affirmations = response_dict.get("affirmations", [])
                return affirmations,messages
            except json.JSONDecodeError as e:
                print("JSON Decode Error: %s", e)
                attempt += 1
                if attempt < max_retries:
                    wait_time = min(1 * (2 ** attempt), 30)  # 指数回退，最大等待30秒
                    print("Retrying in %s seconds...", wait_time)
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        except RateLimitError:
            attempt += 1
            wait_time = min(1 * (2 ** attempt), 30)
            print("Rate limit reached, retrying in %s seconds...", wait_time)
            time.sleep(wait_time)
        except Exception as e:
            print("Unexpected error: %s", e)
            attempt += 1
            if attempt < max_retries:
                wait_time = min(1 * (2 ** attempt), 30)
                print("Retrying in %s seconds...", wait_time)
                time.sleep(wait_time)
            else:
                raise
    return [],[]  # 如果所有重试都失败，返回空列表

def make_Affirmative_by_need(article, need_1, need_2, need, sentences, client, user_problem, zhihu_link, think_log, output_file,role,messages=None):
    max_retries = 3
    retry_delay = 5 
    style = '余华'
    role_prompt = get_role_prompt(role, style=style, articles=article, sentence=sentences, need=need)
    # print(messages)
    messages.append({"role": "user", "content": role_prompt})

    # debug(role_prompt)
    # think_log += clean_value(str(role_prompt))

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-auto",
                messages=messages,
                temperature=1,
                response_format={"type": "json_object"},  # 确保返回 JSON 格式
                n=1  # 请求返回1个结果
            )
            response = completion.choices[0].message.content.strip()
            debug("API Response: ", response)
            # think_log += clean_value(str(response))
            try:
                response_dict = json.loads(response)
                print("Response is valid JSON.")
                if "self_affirmation" in response_dict and isinstance(response_dict["self_affirmation"], list):
                    response_data = response_dict["self_affirmation"]
                else:
                    print("Error: Unexpected API response format.")
                    return
                for i, item in enumerate(response_data):
                    if isinstance(item, dict) and "self_affirmative_phrase" in item:
                        self_affirmative_phrase = item["self_affirmative_phrase"]
                        data_item = make_data_item(
                            user_problem=user_problem,
                            need_1=need_1,
                            need_2=need_2,
                            need=need,
                            self_affirmative_phrase=self_affirmative_phrase,
                            type="3",
                            zhihu_link=zhihu_link,
                            think_log=think_log
                        )
                        
                        save_to_csv(output_file.replace('.csv','_3.csv'), data_item,HEADERS)
                    else:
                        print(f"Error: Invalid item format in response data: {item}")
                
                for sentence in sentences:
                    data_item = make_data_item(
                        user_problem=user_problem,
                        need_1=need_1,
                        need_2=need_2,
                        need=need,
                        self_affirmative_phrase=sentence,
                        type="2",
                        zhihu_link=zhihu_link,
                        think_log=think_log
                    )
                    save_to_csv(output_file.replace('.csv','_2.csv'), data_item,HEADERS)
            except json.JSONDecodeError:
                print("Error: Failed to parse API response as JSON.")
            except KeyError as e:
                print(f"Error: Missing expected key in API response: {e}")
        except RateLimitError:
            if attempt < max_retries - 1:
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = retry_delay * 2
            else:
                print("Max retries reached. Failed to generate affirmations.")
                raise







