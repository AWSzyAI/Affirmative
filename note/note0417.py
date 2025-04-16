import pandas as pd
import sys
import logging
import coloredlogs
from tqdm import tqdm
import concurrent.futures
import os
import json
from threading import Lock, Thread
from datetime import datetime

sys.path.append('../src')
from prompt import get_role_prompt
from kimi_api import client, MODEL_NAME

# 创建日志目录和时间戳文件
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'{timestamp}.log')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台只显示命令提示和进度条（用 print 替代 logging）
coloredlogs.install(level='CRITICAL', logger=logger)

checkpoint_lock = Lock()

def get_content_by_type(row, type):
    if type == '合集':
        try:
            with open('./mod/note-合集.md', 'r', encoding='utf-8') as file:
                note_content = file.read()
        except FileNotFoundError:
            logger.error("note-合集.md 文件未找到")
            return {}
        message = note_content.format(self_affirmation=row['自我肯定语'])
    elif type == '感情状况':
        message = f"""
        自我肯定语: {row['自我肯定语']}
        ...
        """
    elif type == '最近的感觉':
        message = f"""
        自我肯定语: {row['自我肯定语']}
        ...
        """
    return message

def clean_value(value):
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ')
    return value

def update_progress(pbar):
    pbar.update(1)

def get_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    else:
        with open(checkpoint_file, 'w') as f:
            json.dump([], f)
        return []

def update_checkpoint(checkpoint_file, index):
    with checkpoint_lock:
        checkpoint_data = get_checkpoint(checkpoint_file)
        if index not in checkpoint_data:
            checkpoint_data.append(index)
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)

def save_results(row_results, input_file):
    output_file = input_file.replace('.csv', '_result.csv')
    retry_output_file = input_file.replace('.csv', '_result_retry.csv')

    valid_collections = [...]
    valid_relationships = {...}
    valid_feelings = {...}
    valid_causes = {...}

    collection_set = set(row_results.get("合集", []))
    if not collection_set.issubset(valid_collections):
        save_file = retry_output_file
    else:
        relationship_list = row_results.get("感情状况", [])
        feeling_list = row_results.get("最近的感觉", [])
        cause_list = row_results.get("什么让你有这种感觉", [])

        if all(r in valid_relationships for r in relationship_list) and \
           all(f in valid_feelings for f in feeling_list) and \
           all(c in valid_causes for c in cause_list):
            save_file = output_file
        else:
            save_file = retry_output_file

    row_results_df = pd.DataFrame([row_results])
    row_results_df.to_csv(save_file, mode='a', header=not os.path.exists(save_file), index=False)

def get_annotation(row, type='合集'):
    role = f"noter-0205-{type}"
    role_prompt = get_role_prompt(role=role)
    content = get_content_by_type(row, type)
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": content}
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
            n=1
        )
        response = completion.choices[0].message.content.strip()
        logger.warning(f"原始响应内容: {response}")
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}\n内容如下：\n{response}")
            return {}
        result = {}
        if type == "合集":
            result['合集'] = response_dict.get('value', [])
            result['合集理由'] = clean_value(response_dict.get('note', []))
        elif type == "感情状况":
            result['感情状况'] = response_dict.get('value', [])
            result['感情状况理由'] = clean_value(response_dict.get('note', []))
        elif type == "最近的感觉":
            result['最近的感觉'] = response_dict.get("最近的感觉", [])
            result['什么让你有这种感觉'] = response_dict.get('什么让你有这种感觉', [])
            result['最近的感觉理由'] = clean_value(response_dict.get('note', []))
        return result
    except Exception as e:
        logger.error(f"获取{type}标注时发生错误: {e}")
        return {}

def process_row(index, row, checkpoint_file, input_file, pbar):
    row_results = {"自我肯定语": row['自我肯定语']}
    for type in ['合集', '感情状况', '最近的感觉']:
        row_results.update(get_annotation(row, type))
    save_results(row_results, input_file)
    update_checkpoint(checkpoint_file, index)
    Thread(target=update_progress, args=(pbar,)).start()

def load_checkpoint(checkpoint_file):
    return set(get_checkpoint(checkpoint_file))

def process_file(input_file, k=None, from_retry=False, retry_file=None):
    if from_retry and retry_file:
        data = pd.read_csv(retry_file)
    else:
        data = pd.read_csv(input_file)
    if k:
        data = data.sample(k)

    checkpoint_file = input_file.replace('.csv', '_checkpoint.txt')
    completed_indices = load_checkpoint(checkpoint_file)
    print(f"已完成的索引: {completed_indices}")

    with tqdm(total=len(data), initial=len(completed_indices), desc='进度', unit='task') as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for index, row in data.iterrows():
                if index in completed_indices:
                    pbar.update(1)
                    continue
                future = executor.submit(process_row, index, row, checkpoint_file, input_file, pbar)
                futures[future] = index
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"任务{index}失败: {e}")

def main():
    input_file = '../data/0401句子更新 - Sheet1.csv'
    print("选择处理方式：\n1. 从头开始\n2. 从失败列表开始")
    choice = input("请输入选项(1或2): ").strip()
    if choice == "1":
        process_file(input_file, k=500)
    elif choice == "2":
        retry_file = input_file.replace('.csv', '_result_retry.csv')
        process_file(input_file, from_retry=True, retry_file=retry_file)

    output_file = input_file.replace('.csv', '_result.csv')
    retry_output_file = input_file.replace('.csv', '_result_retry.csv')
    completed = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()
    failed = pd.read_csv(retry_output_file) if os.path.exists(retry_output_file) else pd.DataFrame()

    print(f"当前共 {len(completed) + len(failed)} 条，已完成 {len(completed)}，失败 {len(failed)}")
    retry = input("是否立即重试失败部分？(y/n): ").strip().lower()
    if retry == "y":
        process_file(input_file, from_retry=True, retry_file=retry_output_file)
    else:
        print("程序已退出。")

if __name__ == "__main__":
    main()