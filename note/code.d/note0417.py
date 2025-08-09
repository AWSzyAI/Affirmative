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
        根据以下自我肯定语及其相关信息，判断是否适合推送给处于特定感情状况的用户，并根据适合的感情状况返回JSON格式的结果。

        自我肯定语: {row['自我肯定语']}
    

        1. 判断该自我肯定语及其场景描述是否适合以下感情状况的用户：
        ["正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"]。
        2. 选择一个或多个感情状况标签进行标注，**不要**选择其他标签。
        3. 返回的标签列表应该准确反映与该自我肯定语的适配情况。
        4. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

        请以如下格式返回结果：
        {{
            "key": "感情状况",
            "value": ["正在恋爱", "单身但准备好了开始新的恋情"],
            "note": "不放一些标签的理由"
        }}
        """
    elif type == '最近的感觉':
        message = f"""
        请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定情感状态的用户，并根据用户的情感状态返回相应的JSON格式结果。

        自我肯定语: {row['自我肯定语']}

        1. 判断该自我肯定语及其场景描述是否适合推送给以下情感状态的用户：
        ["开心", "很好", "一般", "不好", "糟糕"]。
        2. 请根据句子的情感倾向选择一个或多个标签：  
        - 【开心】和【很好】可以同时出现，  
        - 【不好】和【糟糕】也可以同时出现，  
        - 不要选择额外的标签。
        3. 1级句子倾向于【不好】和【糟糕】，2级句子倾向于【开心】和【很好】，但不是绝对的，具体分析。
        4. "什么让你有这种感觉"是对"最近的感觉"的解释，请从以下选项中选择与感受相关的因素：
        ["家庭", "朋友", "工作", "健康", "感情", "学业", "自己"]。
        5. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

        请以如下格式返回结果：
        {{
            "最近的感觉": ["不好", "糟糕"],
            "什么让你有这种感觉": ["家庭", "健康"],
            "note": "不放一些标签的理由"
        }}
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

    valid_collections = [
        '艰难时刻', '平静内心', '抗击抑郁', '心碎疗愈', '自我关怀', '减压放松', '感恩练习', '爱我的身体', '爱自己',
        '自尊心', '积极心态', '愉悦心情', '焕发生命力', '点燃动力', '自信', '拥抱女性身份', '女性力量', '亲密关系',
        '人际交往', '爱家人', '原谅', '个人成长', '事业成功', '擦拭信念', '追梦无悔','我很好'
    ]
    valid_relationships = {"正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"}
    valid_feelings = {"开心", "很好", "一般", "不好", "糟糕"}
    valid_causes = {"家庭", "朋友", "工作", "健康", "感情", "学业", "自己"}

    collection_set = set(row_results.get("合集", []))
    if not collection_set.issubset(valid_collections):
        print(f"合集不在有效合集列表中: {collection_set}")
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
    # 序号,权重,自我肯定语,曝光点赞率,不感兴趣次数,是否短断句,生产者,合集,合集理由,感情状况,感情状况理由,最近的感觉,什么让你有这种感觉,最近的感觉理由,场景,子场景,用户需求,心理作用机制与功能,句子级别,备注

    row_results = {
        "序号": row['序号'],
        "自我肯定语": row['自我肯定语'],
    }

    note_types = ['合集', '感情状况', '最近的感觉']
    # note_types = ['合集']

    for type in note_types:
        row_results.update(get_annotation(row, type))

    # 序号,自我肯定语,权重,曝光点赞率,不感兴趣次数,是否短断句,生产者,感情状况,感情状况理由,最近的感觉,什么让你有这种感觉,最近的感觉理由,场景,子场景,用户需求,心理作用机制与功能,句子级别,备注,合集,合集理由
    new_columns = [
        "序号", "自我肯定语", '合集', '合集理由',"权重", "曝光点赞率", "不感兴趣次数", "是否短断句", "生产者",
        "感情状况", "感情状况理由", "最近的感觉", "什么让你有这种感觉", "最近的感觉理由",
        "场景", "子场景", "用户需求", "心理作用机制与功能", "句子级别", "备注",
    ]
    row_results = {col: row_results.get(col, '') for col in new_columns}
    
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
    input_file = '../data/全部重标.csv'
    print("选择处理方式：\n1. 从头开始\n2. 从失败列表开始")
    choice = input("请输入选项(1或2): ").strip()
    if choice == "1":
        # process_file(input_file, k=5)
        process_file(input_file)
    elif choice == "2":
        retry_file = input_file.replace('.csv', '_result_retry.csv')
        process_file(input_file, from_retry=True, retry_file=retry_file)

    output_file = input_file.replace('.csv', '_result.csv')
    retry_output_file = input_file.replace('.csv', '_result_retry.csv')
    completed = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()
    failed = pd.read_csv(retry_output_file) if os.path.exists(retry_output_file) else pd.DataFrame()

    print(f"当前共 {len(completed) + len(failed)} 条，已完成 {len(completed)}，失败 {len(failed)}")
    
    if len(failed) == 0:
        print("所有任务已完成！")
        return
    retry = input("是否立即重试失败部分？(y/n): ").strip().lower()
    if retry == "y":
        process_file(input_file, from_retry=True, retry_file=retry_output_file)
    else:
        print("程序已退出。")

if __name__ == "__main__":
    main()