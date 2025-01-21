import csv
import time
import random
from openai import OpenAI
import openai
# from config import KIMI_API_KEY, BASE_URL
from tqdm import tqdm
from openai import RateLimitError  # 导入 RateLimitError 异常
import json
import requests
import os
import concurrent.futures
from prompt import get_role_prompt
from dotenv import load_dotenv

load_dotenv()

KIMI_API_KEY = os.getenv("KIMI_API_KEY")
BASE_URL = "https://api.moonshot.cn/v1"

def load_csv(file_path):
    """加载CSV文件，返回列表格式的数据"""
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def clean_value(value):
    """清理字段中的换行符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ')  # 删除换行符和回车符
    return value

def save_to_csv(output_file, data):
    """将生成的自我肯定语及其对应数据保存到CSV文件"""
    with open(output_file, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件为空，则写入表头
        if file.tell() == 0:
            writer.writerow(['用户问题/症状', '子场景症状合并', '标签（附加参考，用于引导生成或校正句子内容）', '自我肯定语','参考句子1', '参考句子2', '参考句子3', '参考句子4', '参考句子5','参考句子6', '参考句子7', '参考句子8', '参考句子9', '参考句子10','参考句子11', '参考句子12', '参考句子13', '参考句子14', '参考句子15','参考句子16', '参考句子17', '参考句子18', '参考句子19', '参考句子20'])
        
        # 写入每行数据
        for row in data:
            cleaned_row = [clean_value(row[field]) for field in ['用户问题/症状', '子场景症状合并', '标签（附加参考，用于引导生成或校正句子内容）', '自我肯定语', '参考句子1', '参考句子2', '参考句子3', '参考句子4', '参考句子5','参考句子6', '参考句子7', '参考句子8', '参考句子9', '参考句子10','参考句子11', '参考句子12', '参考句子13', '参考句子14', '参考句子15','参考句子16', '参考句子17', '参考句子18', '参考句子19', '参考句子20']]
            writer.writerow(cleaned_row)
            # writer.writerow([row['用户问题/症状'], row['子场景症状合并'], row['标签（附加参考，用于引导生成或校正句子内容）'], row['自我肯定语'],row['参考句子1'],row['参考句子2'],row['参考句子3'],row['参考句子4'],row['参考句子5']])

def get_checkpoint(checkpoint_file):
    """从 checkpoint 文件中获取已生成的用户问题/症状的索引"""
    try:
        with open(checkpoint_file, 'r') as f:
            last_processed_index = int(f.read().strip())  # 读取最后处理的索引
    except (FileNotFoundError, ValueError):
        last_processed_index = 0  # 如果没有文件或文件内容不合法，默认从第0个开始
    return last_processed_index

def update_checkpoint(checkpoint_file, index):
    """更新 checkpoint 文件"""
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))  # 记录当前处理到的索引



def get_encouragements(message,k=5):
    # 利用message 检索鼓励语quote   
    # encouragement_quotes = get_euotes_k(message,k)
    
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
        print("请求失败:", response.status_code, response.text)
    
    print(encouragement_quotes)
    return encouragement_quotes


def generate_self_affirmative_phrase_concurrent(symptoms_data, csv_file, checkpoint_file, n=5, delay=0.5, max_retries=5):
    """生成n条不相似的自我肯定语，添加延时以防止触发RateLimit，并保存结果到CSV，使用并发"""
    client = OpenAI(api_key=KIMI_API_KEY, base_url=BASE_URL)
    result_data = []

    # 获取已生成的用户问题/症状，避免重复生成
    last_processed_index = get_checkpoint(checkpoint_file)

    # 使用 tqdm 来显示生成进度条
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        # 将每个任务提交到线程池
        for i in range(last_processed_index, len(symptoms_data)):
            symptom = symptoms_data[i]
            user_problem = symptom['用户问题/症状']
            additional_info = symptom['标签（附加参考，用于引导生成或校正句子内容）']

            # 构造生成任务
            futures.append(executor.submit(generate_affirmation_for_symptom, i, symptom, user_problem, additional_info, client, n, delay, max_retries, result_data, csv_file, checkpoint_file))

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 捕获异常，如果任务有异常，会抛出

    print(f"所有未生成过的自我肯定语已保存到 {csv_file}")
    # 删除checkpoint_file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"已删除文件: {checkpoint_file}")
    else:
        print(f"文件不存在: {checkpoint_file}")




query_text = '失恋-悲观（觉得自己遇不到更好的了）'

def query_article(query_text,top_k = 2):
    try:
        query_vector = embeddings.embed_query(query_text)
        article_data = query_article_data('article_collection', query_vector, top_k)
        print(article_data)
        # return jsonify({"message": "查询成功", "data": article_data})
    except Exception as e:
        print(e)
        # return jsonify({"message": "查询失败", "error": str(e)})   



def generate_affirmation_for_symptom(i, symptom, user_problem, additional_info, client, n, delay, max_retries, result_data, csv_file, checkpoint_file):
    """生成单个症状的自我肯定语并保存到 CSV"""
    message = f"症状: {user_problem}\n附加信息: {additional_info}"
    encouragement_quotes = get_encouragements(message, 20)
    articles = query_article(message,2)
    role_prompt = get_role_prompt("productor", init=encouragement_quotes,articles = articles)

    # 重试机制
    attempt = 0
    while attempt < max_retries:
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[{"role": "system", "content": role_prompt},
                          {"role": "user", "content": message}],
                temperature=1,
                response_format={"type": "json_object"},  # JSON Mode
                n=1  # 请求返回1个结果，因为返回的是JSON格式
            )
            response = completion.choices[0].message.content.strip()
            # print("API Response:", response)  # 打印响应内容
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print("Response is not valid JSON:", response)
                continue  # 这里要修改：重新request一遍，而不是continue

            think_log = response_data.get("think_log", "无思考日志")  # 获取思考日志
            # print(f"思考日志: {think_log}")  # 打印或保存思考日志，便于分析



            for item in response_data["results"]:
                self_affirmative_phrase = item["self_affirmative_phrase"]

                # 将生成的结果保存到CSV文件
                result_data.append({
                    '用户问题/症状': user_problem,
                    '子场景症状合并': symptom['子场景症状合并'],
                    '标签（附加参考，用于引导生成或校正句子内容）': additional_info,
                    '自我肯定语': self_affirmative_phrase,
                    '参考句子1': encouragement_quotes[0],
                    '参考句子2': encouragement_quotes[1],
                    '参考句子3': encouragement_quotes[2],
                    '参考句子4': encouragement_quotes[3],
                    '参考句子5': encouragement_quotes[4],
                    '参考句子6': encouragement_quotes[5],
                    '参考句子7': encouragement_quotes[6],
                    '参考句子8': encouragement_quotes[7],
                    '参考句子9': encouragement_quotes[8],
                    '参考句子10': encouragement_quotes[9],
                    '参考句子11': encouragement_quotes[10],
                    '参考句子12': encouragement_quotes[11],
                    '参考句子13': encouragement_quotes[12],
                    '参考句子14': encouragement_quotes[13],
                    '参考句子15': encouragement_quotes[14],
                    '参考句子16': encouragement_quotes[15],
                    '参考句子17': encouragement_quotes[16],
                    '参考句子18': encouragement_quotes[17],
                    '参考句子19': encouragement_quotes[18],
                    '参考句子20': encouragement_quotes[19],
                })

                # 每生成一个自我肯定语后立即保存
                save_to_csv(csv_file, result_data)
                result_data.clear()  # 清空当前生成的结果，避免重复保存

            # 更新 checkpoint 文件，记录最后处理到的索引
            update_checkpoint(checkpoint_file, i + 1)
            break  # 如果成功，则跳出重试循环
        except RateLimitError:
            # 增加延迟时间，并进行重试
            attempt += 1
            wait_time = delay * (2 ** attempt)  # 指数回退：1, 2, 4, 8...
            print(f"Rate limit reached, retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # 等待一段时间后重试
