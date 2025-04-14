
import os
import csv
import time
import json
import requests
import concurrent
import re
import threading

from tqdm import tqdm
from openai import RateLimitError  # 导入 RateLimitError 异常
# from src.deepseek_api import client,MODEL_NAME,send_messages # Deepseek
# from src.milvus_utils import embeddings, query_article_data
# from src.kimi_api import client,MODEL_NAME,send_messages # kimi
# from src.prompt import get_role_prompt,get_paradigm

import sys
import os

# 获取项目根目录的绝对路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 添加 src 目录到 sys.path 中
sys.path.append(os.path.join(root_dir, 'src'))
from milvus_utils import embeddings, query_article_data
# from kimi_api import client,MODEL_NAME,send_messages # kimi
from ark_api import client,MODEL_NAME,send_messages # doubao
from prompt import get_role_prompt,get_paradigm
from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_attempt,before_sleep_log
import logging
from loguru import logger
import coloredlogs
import openai
import ast
from datetime import datetime
from pathlib import Path


DEBUG = True

HEADERS = ['自我肯定语','句子范式','role','model','生产者', '场景','子场景','场景描述','用户需求','心理作用机制与功能','句子级别', 'zhihu_link']
HEADERS_structured_article = ['发问：思考、反省', '价值观', '行动：可效仿的行动指南', '慈悲：理解、接受、宽恕', '状态描述：成为这样的我']
checkpoint_lock = threading.Lock()  # 线程锁，用于保护检查点文件的更新
BAN_WORDS = ["子女", "女儿", "儿子", "孩子","防疫", "3D", "儿童", "幼儿", "幼年", "父母", "妈妈","情绪","疫情"]
# def debug(*args, **kwargs):
#     if DEBUG:
#         print(*args, **kwargs)
paradigms = [
    '情绪应对式: 简单-情绪应对式',
    # 认知行为-情绪应对式
    # 正念-情绪应对式
    # 道家-情绪应对式
    '安抚接纳式: 简单-自我接纳式',
    '安抚接纳式: 简单-环境接纳式',
    '安抚接纳式: 人本主义-自我接纳式',
    '安抚接纳式: 积极心理-自我接纳式',
    "外源锚定式: 简单-外源锚定式",
    "心态稳定式: 简单-心态稳定式",
    '积极感知式: 简单-积极感知式',
    '主体自信式: 简单-主体自信式',
    '主体自信式: 他者-主体自信式',
    '潜能确认式: 简单-潜能承认式',
    "心念宣告式: 简单-心念宣告式",
    '行动宣告式: 简单-行动宣告式',
    "动态改变式: 简单-动态改变式",
    '意义构建式: 简单-主体意义式',
    '意义构建式: 简单-经历意义式',
    "感恩整合式: 简单-感恩整合式",
    '主权宣告式: 简单-主权宣告式',
    '主权宣告式: 逻辑-主权宣告式',
    "独特价值宣言式: 简单-独特价值宣言式",
    "对抗超升式: 权力意志-对抗超升式",
    ]

# matched_paradigms = paradigms_matcher(paradigms,symptom)
matched_paradigms = [
    '情绪应对式: 简单-情绪应对式',
    "情绪应对式: 认知行为-情绪应对式",
    "情绪应对式: 正念-情绪应对式",
    "情绪应对式: 道家-情绪应对式",
    '安抚接纳式: 简单-自我接纳式',
    '安抚接纳式: 简单-环境接纳式',
    # "安抚接纳式: 人本主义-自我接纳式",
    # "安抚接纳式: 积极心理-自我接纳式",
    "外源锚定式: 简单-外源锚定式",
    "心态稳定式: 简单-心态稳定式",
    '积极感知式: 简单-积极感知式',
    '主体自信式: 简单-主体自信式',
    '主体自信式: 逻辑-主体自信式',
    # "主体自信式: 他者-主体自信式",
    '潜能确认式: 简单-潜能确认式',
    "心念成长式: 简单-心念培育式",
    "心念成长式: 逻辑-心念培育式",
    "心念成长式: 简单-心念锚定式",
    "心念成长式: 逻辑-心念锚定式",
    "行动宣告式: 简单-行动宣告式",
    "自然改变式: 简单-自然改变式",
    "意义构建式: 简单-主体意义式",
    "意义构建式: 简单-经历意义式",
    "价值锚定式: 简单-价值锚定式",
    "价值锚定式: 逻辑-价值锚定式",
    "感恩整合式: 简单-感恩整合式",
    "独特价值宣言式: 简单-独特价值宣言式",
    "主权宣告式: 简单-主权宣告式",
    # "主权宣告式: 逻辑-主权宣告式",
    "对抗超升式: 权力意志-对抗超升式",
    "爱之循环式: 简单-爱之循环式",
    ]
max_retries = 5



def debug(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)  # 确保所有参数都转换为字符串
    logger.debug(message, **kwargs)

# 切换日志级别的函数
def set_log_level(level: str):
    level = level.upper()
    logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    coloredlogs.install(level='DEBUG', logger=logger)
    # if level == 'DEBUG':
    #     logger.setLevel(logging.DEBUG)
    #     file_handler.setLevel(logging.DEBUG)
    #     console_handler.setLevel(logging.DEBUG)
    #     coloredlogs.install(level='DEBUG', logger=logger)
    # else:
    #     logger.setLevel(logging.INFO)
    #     file_handler.setLevel(logging.INFO)
    #     console_handler.setLevel(logging.INFO)
    #     coloredlogs.install(level='INFO', logger=logger)

# set_log_level("DEBUG")  # 切换到DEBUG模式
# set_log_level("INFO")   # 切换回INFO模式
    


# FILE I/O
def make_data_item(type,symptom,self_affirmative_phrase=None,user_problem=None, need_1=None, need_2=None, need=None, structured_articles=None,zhihu_link=None, think_log=None,role=None,model=None):
    """构造数据项"""
    if type == 'structured_article':
        return {
            '发问：思考、反省':structured_articles.get('发问：思考、反省', 'N/A'),
            '价值观':structured_articles.get('价值观', 'N/A'),
            '行动：可效仿的行动指南':structured_articles.get('行动：可效仿的行动指南', 'N/A'),
            '慈悲：理解、接受、宽恕':structured_articles.get('慈悲：理解、接受、宽恕', 'N/A'),
            '状态描述：成为这样的我':structured_articles.get('状态描述：成为这样的我', 'N/A')
        }
    elif type == '0203-3':
        return {
            '场景': symptom['场景'],
            '子场景': symptom['子场景'],
            '场景描述': symptom['场景描述'],
            '用户需求': symptom['用户需求'],
            '心理作用机制与功能': symptom['心理作用机制与功能'],
            '句子级别': symptom['句子级别'],
            '句子范式': symptom['句子范式'],
            '自我肯定语': self_affirmative_phrase,
            '生产者':type,
            'zhihu_link':zhihu_link,
            'role':role,
            'model':model
        }
    elif type=='0203-2':
        return {
            '场景': symptom['场景'],
            '子场景': symptom['子场景'],
            '场景描述': symptom['场景描述'],
            '用户需求': symptom['用户需求'],
            '心理作用机制与功能': symptom['心理作用机制与功能'],
            '句子级别': symptom['句子级别'],
            '自我肯定语': self_affirmative_phrase,
            '生产者':type,
            'zhihu_link':zhihu_link,
            'role':role,
            'model':model
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
# def get_encouragements(message, k=5):
#     """利用 message 检索鼓励语 quote"""
#     encouragement_url = "http://test.caritas.pro:5001/query_excerpt_data"
#     payload = {
#         "query_text": message,
#         "top_k": k
#     }
#     headers = {
#         "Content-Type": "application/json"
#     }
#     response = requests.post(encouragement_url, json=payload, headers=headers)
#     encouragement_quotes = []
#     if response.status_code == 200:
#         data = response.json()
#         for item in data["data"]:
#             encouragement_quotes.append(item["entity"]["quote"])
#     else:
#         print("请求失败: %s %s", response.status_code, response.text)
#     return encouragement_quotes

def query_article(query_text, top_k=2):
    """查询文章数据"""
    # debug(f"{query_text}")
    try:
        query_vector = embeddings.embed_query(query_text)
        article_data = query_article_data('article_collection', query_vector, top_k)
        # debug(article_data)
        # debug(f"{query_text} Done")

        # 过滤掉包含 "想法集" 标签的文章
        filtered_article_data = [
            article for article in article_data 
            if "想法集" not in article['entity']['tags']
        ]
        if not filtered_article_data:
            print('未检索到articles')
            return []
        # if len(filtered_article_data) < len(article_data):
            # debug("Found '想法集' tag, removed from results.")
        return filtered_article_data
    except Exception as e:
        print("Error: %s", e)
        return []



@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避，初始4秒，最大60秒
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(max_retries),  # 使用函数参数控制最大重试次数
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def generate_affirmation_for_symptom_with_retry(*args, **kwargs):
    """带重试机制的API调用包装函数"""
    return generate_affirmation_for_symptom(*args, **kwargs)

def update_progress(pbar):
    pbar.update(1)

# main Job
def generate_self_affirmative_phrase_concurrent(
        symptoms_file, 
        csv_file, 
        checkpoint_file,
        paradigm_md_path, 
        n, 
        delay, 
        max_retries,
        DEBUG_model, 
        max_length,
        use_concurrency=False,
        timeout=1800,
        log_file=None 
    ):
    
    global logger,file_handler,console_handler

    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器并设置日志文件路径

    # log_file = './Log/'+datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log'
    print(log_file)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建控制台输出处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    coloredlogs.install(level='INFO', logger=logger)



    global DEBUG
    if DEBUG_model==True:
        print("DEBUG模式开启")
        set_log_level("DEBUG") 
        DEBUG=True
    else:
        set_log_level("INFO")
        DEBUG=False
    # 在代码中记录调试日志
    if DEBUG:
        logger.debug("This is a debug message with detailed info.")
    else:
        logger.info("This is an info message.")
    symptoms_data = load_csv(symptoms_file)
    completed_indices = set(get_checkpoint(checkpoint_file)) 
    print(f"从检查点文件读取到已完成的索引: {completed_indices}")
    
    with tqdm(total=len(symptoms_data), initial=len(completed_indices), desc="生成进度", unit="item", position=0) as pbar:
        if use_concurrency:
            MAX_CONCURRENT_WORKERS = 3
            # 并发执行逻辑
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
                futures = {}
                for i in range(len(symptoms_data)):
                    if i in completed_indices:  # 如果任务已完成，跳过
                        pbar.update(1)  # 更新进度条
                        continue
                    # print(symptoms_data[i])
                    
                    future = executor.submit(
                        generate_affirmation_for_symptom, i, symptoms_data[i], n, delay, max_retries, csv_file, paradigm_md_path,max_length=max_length, DEBUG=DEBUG
                    )
                    futures[future] = i  # 将 future 和索引关联起来


                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]  # 获取当前任务的索引
                    try:
                        future.result()  # 捕获异常，如果任务有异常，会抛出
                        threading.Thread(target=update_progress, args=(pbar,)).start()
                        update_checkpoint(checkpoint_file, index)  # 更新检查点文件
                    except Exception as e:
                        if DEBUG:
                            raise
                        print(f"任务 {index} 失败: {e}")
                    
                        
        else:
            # 串行执行逻辑
            for i in range(len(symptoms_data)):
                if i in completed_indices:  # 如果任务已完成，跳过
                    pbar.update(1)  # 更新进度条
                    continue
                # print(symptoms_data[i])
                try:
                    
                    
                    generate_affirmation_for_symptom(i, symptoms_data[i], n, delay, max_retries, csv_file ,paradigm_md_path,  max_length=max_length,DEBUG=DEBUG)
                    
                    
                    pbar.update(1)  # 更新进度条
                    update_checkpoint(checkpoint_file, i)  # 更新检查点文件
                except Exception as e:
                    print(f"任务 {i} 失败: {e}")
                    if DEBUG:
                        raise
    print(f"所有未生成过的自我肯定语已保存到 {csv_file.replace('.csv','_*.csv')}")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"已删除文件: {checkpoint_file}")
    else:
        print(f"文件不存在: {checkpoint_file}")


def extract_json(response):
    # 使用正则表达式匹配最外层 {} 之间的内容，确保返回的是 JSON 格式
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        return match.group(0)  # 提取整个 JSON 部分
    else:
        return response  # 如果没有匹配到，返回原始内容
#  pipeline LLM API
def get_structured_articles(article_data, role):
    """
    对article_data中对每一篇article的content进行结构化转换
    """
    max_retries = 3 if not DEBUG else 1 # 调试时减少重试次数
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
                messages=[
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": content}
                ]
                
                # completion = client.chat.completions.create(
                #     model=MODEL_NAME,
                #     messages=messages,
                #     temperature=1, #kimi
                #     # temperature=1.3, #deepseek
                #     response_format={"type": "json_object"},  # 确保返回 JSON 格式
                #     n=1  # 请求返回1个结果
                # )
                # debug(completion)
                # 解析 API 返回的内容
                # response = completion.choices[0].message.content.strip()
                
                response = send_messages(messages)
                # debug(response) # ok
                # # 如果response 是```json {*}```,就去掉{*}之外的内容
                response = extract_json(response)
                debug(response) # ok
                structured_article = json.loads(response)  # 将字符串解析为 JSON 对象
                structured_articles.append(structured_article)
                break  # 成功后跳出循环
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Failed to generate structured article.")
                    raise
                if DEBUG:
                    raise # 调试时直接抛出异常

        # 添加延时，降低请求速率
        time.sleep(1)  # 每次请求后延时 1 秒

    return structured_articles

def make_Affirmative(role,symptom,content,articles,messages):
    max_retries = 3
    if not content:
        return [" "]

    role_prompt = get_role_prompt(role,articles=articles)
    message = "素材句子（核心内容、含义）："+"\n".join(content)+f"\n 文章”“”{articles}“”“，生成10个句子，句子级别为{symptom['句子级别']}级"
    
    # 如果messages为空（None 或 空列表[]），则初始化
    if not messages:  # 这种判断既能捕捉 None 也能捕捉空列表 []
        messages = [{"role": "system", "content": role_prompt}, {"role": "user", "content": message}]
    else:
        # 如果messages已经有内容，追加新的消息
        messages.append({"role": "user", "content": message})
    
    attempt = 0
    while attempt < max_retries:
        try:       
            # completion = client.chat.completions.create(
            #     model=MODEL_NAME,
            #     messages = messages,
            #     # temperature=1.3, #deepseek
            #     temperature=1, #kimi
            #     response_format={"type": "json_object"},  # 确保返回 JSON 格式
            #     n=1  # 请求返回1个结果
            # )
            # response = completion.choices[0].message.content.strip()

            
            # print(completion.choices[0])
            response = send_messages(messages)
            # debug(response) # ok
            # # 如果response 是```json {*}```,就去掉{*}之外的内容
            response = extract_json(response)
            # debug(response) # ok
            try:
                response_dict = json.loads(response)
                affirmations = response_dict.get("affirmations", [])
                return affirmations,messages
            except json.JSONDecodeError as e:
                debug(messages)
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
            if DEBUG:
                raise
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

def make_Affirmative_by_need(symptom,paradigm, sentences,zhihu_link, output_file,paradigm_md_path,messages=None):
    # 忽略2号员工的历史对话
    # messages = []

    max_retries = 3
    retry_delay = 5 
    
    # style = '余华'
    # role_prompt = get_role_prompt(role, max_length=max_length,symptom=symptom, style=style, articles=article, sentence=sentences)
    # messages.append({"role": "user", "content": role_prompt})

    paradigm_prompt = get_paradigm(paradigm,symptom=symptom,sentences=sentences,paradigm_md_path = paradigm_md_path)
    
    # file_object = client.files.create(file=Path("/home/acszy/2025/Affirmative/data/艾里希·弗洛姆三部曲（社会心理学大师经典著作，爱的艺术+论不服从+存在的艺术） (艾里希·弗洛姆 [艾里希·弗洛姆]) (Z-Library).txt"), purpose="file-extract")
    # file_object = client.files.create(file=Path("/home/acszy/2025/Affirmative/data/book_1.txt"), purpose="file-extract")
    # file_content_bytes = client.files.content(file_id=file_object.id).content  # 获取二进制内容
    # file_content = file_content_bytes.decode("utf-8")  # 以 UTF-8 解码
    # messages.append({"role": "system","content": "用这本书的内容作为素材来生成句子："+file_content})
    messages.append({"role": "user", "content": paradigm_prompt})
    debug(messages)
    # debug(role_prompt)
    # think_log += clean_value(str(role_prompt))

    for attempt in range(max_retries):
        try:
            response = extract_json(send_messages(messages))
            debug("API Response: ", response)
            
            try:
                response_dict = json.loads(response)
                # debug("Response is valid JSON.")
                if "self_affirmation" in response_dict and isinstance(response_dict["self_affirmation"], list):
                    response_data = response_dict["self_affirmation"]
                    # debug("API Response: ", response)
                else:
                    print("Error: Unexpected API response format.")
                    # debug("API Response: ", response)
                    return
                
                for i, item in enumerate(response_data):
                    if not (isinstance(item, dict) and "self_affirmative_phrase" in item):
                        print(f"Error: Invalid item format in response data: {item}")
                        return
                    self_affirmative_phrase = item["self_affirmative_phrase"]
                    if any(word in self_affirmative_phrase for word in BAN_WORDS):
                        continue
                    type="0203-3"
                    data_item = make_data_item(
                        self_affirmative_phrase=self_affirmative_phrase,
                        type=type,
                        symptom=symptom,
                        zhihu_link=zhihu_link,
                        role = paradigm,
                        model = MODEL_NAME
                        )
                    # print(output_file.replace('.csv','_3.csv'))
                    save_to_csv(output_file.replace('.csv','_3.csv'), data_item,HEADERS)                
                return 
            
            except json.JSONDecodeError:
                print("Error: Failed to parse API response as JSON.")
                if DEBUG:
                    raise
                
            except KeyError as e:
                print(f"Error: Missing expected key in API response: {e}")
                if DEBUG:
                    raise
                
        except openai.OpenAIError as e:
            print(f"OpenAIError: {e}")
            # Log or handle the error as needed
            # if DEBUG:
            print("Problematic prompt:", messages)
        except RateLimitError:
            
            if attempt < max_retries - 1:
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = retry_delay * 2
            else:
                print("Max retries reached. Failed to generate affirmations.")
            if DEBUG:
                raise


def generate_affirmation_for_symptom(i, symptom, n, delay, max_retries, csv_file,paradigm_md_path, max_length,DEBUG=False):

    # for paradigm in matched_paradigms:
    #     # print(f"原始 symptom['句子范式']: {repr(symptom['句子范式'])}")  
    #     # print(f"paradigm: {repr(paradigm)}")  

    #     # 1️⃣ 处理 symptom['句子范式']，确保它是一个字符串并去掉引号
    #     if isinstance(symptom['句子范式'], str):
    #         symptom_cleaned = symptom['句子范式'].strip('"').strip("'").strip("[]")  # 去掉外部的引号和 []
    #         symptom_list = symptom_cleaned.replace('，', ',').split(',')  # 兼容全角逗号并拆分
    #         symptom_list = [s.strip() for s in symptom_list]  # 去除前后空格
    #     else:
    #         symptom_list = symptom['句子范式']  # 如果已经是 list，直接使用

    #     # print(f"转换后的 symptom_list: {symptom_list}")

    #     # paradigm = "行动宣告式: 简单-行动宣告式"
    #     query_paradigm = paradigm.split(":", 1)[-1].strip()
    #     # print(query_paradigm)  # 输出：简单-行动宣告式


    #     # 2️⃣ 检查 paradigm 是否在 symptom_list 里
    #     if query_paradigm in symptom_list:
    #         print(f"✅ paradigm '{paradigm}' 匹配成功！")
    #         make_Affirmative_by_need(symptom, paradigm, zhihu_link, csv_file, messages=messages)
    #     else:
    #         print(f"❌ paradigm '{paradigm}' 未匹配！")

    # return


    """生成单个症状的自我肯定语并保存到 CSV"""
    
    # INPUT_HEADERS = ['场景','子场景','场景描述','用户需求', '心理作用机制与功能','句子级别']
    # QUERY_HEADERS = ['子场景', '用户需求', '心理作用机制与功能']
    QUERY_HEADERS = ['子场景']
    
    query_method = 2
    if query_method == 1: # 拼起来检索
        query_message = ' '.join([symptom[keyword] for keyword in QUERY_HEADERS])
        article_data = query_article(query_message, 1)
    elif query_method == 2: # 单独检索
        article_data = []    
        for keyword in QUERY_HEADERS:
            debug(f"query{symptom[keyword]}")
            article_data += query_article(symptom[keyword], 1)
        unique_article_data = remove_duplicates(article_data)
        debug(f"article去重：{len(article_data)} -> {len(unique_article_data)}")
        article_data = unique_article_data # 缺少这一步会导致重复
    print(article_data)
    # # query_book
    # book_PATH = "/home/acszy/2025/Affirmative/data/book_1.txt"
    # # 建立向量数据库
    # book_embeddings = embeddings.embed_documents(book_PATH)
    # # 将向量数据库保存到文件
    # with open('book_embeddings.pkl', 'wb') as f:
    #     pickle.dump(book_embeddings, f)

    # # 从文件中加载向量数据库
    # with open('book_embeddings.pkl', 'rb') as f:
    #     book_embeddings = pickle.load(f)
    
    # # 使用向量数据库进行查询
    # query_vector = embeddings.embed_query(symptom['用户需求'])
    # book_data = 




    zhihu_link = ' '.join([article['entity']['zhihu_link'] for article in article_data]) if article_data else "无链接"
    debug(zhihu_link)
    articles = ' '.join([article['entity']['content'] for article in article_data])

    structured_articles = get_structured_articles(article_data,"article-structurer")
    debug(structured_articles)
    sentences = []
    messages = []
    role_maker = "Affirmative_maker-0213"
    for i, structured_article in enumerate(structured_articles):
        # debug('发问：思考、反省', structured_article.get('发问：思考、反省', 'N/A'))
        # debug('价值观', structured_article.get('价值观', 'N/A'))
        # debug('行动：可效仿的行动指南', structured_article.get('行动：可效仿的行动指南', 'N/A'))
        # debug('慈悲：理解、接受、宽恕', structured_article.get('慈悲：理解、接受、宽恕', 'N/A'))
        # debug('状态描述：成为这样的我', structured_article.get('状态描述：成为这样的我', 'N/A'))
        for j in ['状态描述：成为这样的我','发问：思考、反省','价值观','行动：可效仿的行动指南','慈悲：理解、接受、宽恕']:
            if structured_article.get(j):
                # affirmative,messages = make_Affirmative("Affirmative_maker",symptom,structured_article.get(j),articles=articles,messages=messages)
                affirmative,messages = make_Affirmative(role_maker,symptom,structured_article.get(j),articles=articles,messages=messages)
                sentences.extend(affirmative)
            else:
                print(f"{j} not found")
        # save_to_csv
        structured_item = make_data_item(
            type='structured_article', structured_articles=structured_article,
            symptom=symptom,
            )
        save_to_csv(csv_file.replace('.csv','_structured.csv'), structured_item, HEADERS_structured_article)

    # print("sentences:", sentences)
    debug(f"len(sentences): {len(sentences)}")
    # 去重
    sentences = list(set(sentences))
    if len(sentences) == 0:
        logging.warning(f"No sentences generated for symptom index {i}")
    else:
        sentences = list(set(sentences))
        for sentence in sentences:
            if any(word in sentence for word in BAN_WORDS):
                continue
            # debug(sentence)
            type="0203-2"
            data_item = make_data_item(
                self_affirmative_phrase=sentence,
                type=type,
                symptom=symptom,
                zhihu_link=zhihu_link,
                role = role_maker,
                model = MODEL_NAME
                )
            save_to_csv(csv_file.replace('.csv','_2.csv'), data_item,HEADERS)
    # make_Affirmative_by_need(symptom, articles, sentences, zhihu_link, csv_file,"style-fliter-0204",messages=messages,max_length=max_length)
    # 根据symptom判断范式
    
    # ALL=1
    ALL=0

    # ?  [简单-情绪应对式］都扩展为[简单-情绪应对式,认知行为-情绪应对式,正念-情绪应对式,道家-情绪应对式]
    
    for paradigm in matched_paradigms:
        
        if ALL==1:
            make_Affirmative_by_need(symptom, paradigm, sentences, zhihu_link, csv_file, paradigm_md_path,messages=messages)
            return 
        
        # print(f"原始 symptom['句子范式']: {repr(symptom['句子范式'])}")  
        # print(f"paradigm: {repr(paradigm)}")  

        # 1️⃣ 处理 symptom['句子范式']，确保它是一个字符串并去掉引号
        if isinstance(symptom['句子范式'], str):
            symptom_cleaned = symptom['句子范式'].strip('"').strip("'").strip("[]")  # 去掉外部的引号和 []
            symptom_list = symptom_cleaned.replace('，', ',').split(',')  # 兼容全角逗号并拆分
            symptom_list = [s.strip() for s in symptom_list]  # 去除前后空格
        else:
            symptom_list = symptom['句子范式']  # 如果已经是 list，直接使用

        # print(f"转换后的 symptom_list: {symptom_list}")

        # paradigm = "行动宣告式: 简单-行动宣告式"
        query_paradigm = paradigm.split(":", 1)[-1].strip()
        # print(query_paradigm)  # 输出：简单-行动宣告式


        # 2️⃣ 检查 paradigm 是否在 symptom_list 里
        if query_paradigm in symptom_list:
            # print(f"✅ paradigm '{paradigm}' 匹配成功！")
            make_Affirmative_by_need(symptom, paradigm, sentences, zhihu_link, csv_file, paradigm_md_path,messages=messages)
        else:
            # print(f"❌ paradigm '{paradigm}' 未匹配！")
            pass

    # for paradigm in matched_paradigms:
    #     print(symptom['句子范式'])  # 例如 [简单-行动宣告式,简单-自然改变式，简单-爱之循环式] 不是一个标准的list，仅仅是字符串，应该先加上引号
    #     print(paradigm)  # 例如 "潜能确认式: 简单-潜能确认式"
    #     if paradigm in ast.literal_eval(symptom['句子范式']):
    #         print(f"paradigm {paradigm} matched")        
    #     # make_Affirmative_by_need(symptom, paradigm, zhihu_link, csv_file, messages=messages)