import pandas as pd
import sys
import logging
import coloredlogs
from tqdm import tqdm
import concurrent.futures
import os
import json
from threading import Lock, Thread

sys.path.append('../src')
from prompt import get_role_prompt
from kimi_api import client, MODEL_NAME

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器并设置日志文件路径
log_file = 'app.log'
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
coloredlogs.install(level='INFO', logger=logger)

checkpoint_lock = Lock()

def get_content_by_type(row, type):
    if type == '合集':
        # 读取 note-合集.md 文件内容
        try:
            with open('./mod/note-合集.md', 'r', encoding='utf-8') as file:
                note_content = file.read()
        except FileNotFoundError:
            logger.error("note-合集.md 文件未找到")
            return {}

        # 格式化传参，将自我肯定语插入到文件内容的适当位置
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
    """清理字段中的换行符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ')  # 删除换行符和回车符
    return value
def update_progress(pbar):
    """更新进度条"""
    pbar.update(1)

def get_checkpoint(checkpoint_file):
    """获取 checkpoint 文件中记录的所有已完成索引"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        return checkpoint_data
    else:
        # 如果 checkpoint 文件不存在，初始化为空列表
        with open(checkpoint_file, 'w') as f:
            json.dump([], f)
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

def save_results(row_results, input_file):
    """保存处理结果到文件"""
    output_file = input_file.replace('.csv', '_result.csv')
    retry_output_file = input_file.replace('.csv', '_result_retry.csv')

    # 允许的值
    valid_collections = [
        '艰难时刻', '平静内心', '抗击抑郁', '心碎疗愈', '自我关怀', '减压放松', '感恩练习', '爱我的身体', '爱自己',
        '自尊心', '积极心态', '愉悦心情', '焕发生命力', '点燃动力', '自信', '拥抱女性身份', '女性力量', '亲密关系',
        '人际交往', '爱家人', '原谅', '个人成长', '事业成功', '擦拭信念', '追梦无悔', '勇气'
    ]
    valid_relationships = {"正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"}
    valid_feelings = {"开心", "很好", "一般", "不好", "糟糕"}
    valid_causes = {"家庭", "朋友", "工作", "健康", "感情", "学业", "自己"}

    # 确保 "合集" 是合法子集
    collection_set = set(row_results.get("合集", []))  # "合集" 是一个列表
    if not collection_set.issubset(valid_collections):
        save_file = retry_output_file
    else:
        # 确保 "感情状况" 中的每个值都在允许的集合中
        relationship_list = row_results.get("感情状况", [])
        feeling_list = row_results.get("最近的感觉", [])
        cause_list = row_results.get("什么让你有这种感觉", [])

        relationship_valid = all(r in valid_relationships for r in relationship_list)
        feeling_valid = all(f in valid_feelings for f in feeling_list)
        cause_valid = all(c in valid_causes for c in cause_list)

        # 只要有一个不符合，就存入 retry 文件
        if not (relationship_valid and feeling_valid and cause_valid):
            save_file = retry_output_file
        else:
            save_file = output_file

    row_results_df = pd.DataFrame([row_results])
    if os.path.exists(save_file):
        row_results_df.to_csv(save_file, mode='a', header=False, index=False)
    else:
        row_results_df.to_csv(save_file, mode='a', header=True, index=False)

# def get_annotation(row, type='合集'):
#     role = f"noter-0205-{type}"
#     role_prompt = get_role_prompt(role=role)
    
#     # 根据不同类型生成提示词并返回相应的标注结果
#     content = get_content_by_type(row, type)
#     messages = [
#         {"role": "system", "content": role_prompt},
#         {"role": "user", "content": content}
#     ]
    
#     try:
#         completion = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=messages,
#             temperature=1,
#             response_format={"type": "json_object"},  # 确保返回 JSON 格式
#             n=1  # 请求返回1个结果
#         )
#         response = completion.choices[0].message.content.strip()
#         response_dict = json.loads(response)
        
#         result = {}
#         if type == "合集":
#             result['合集'] = response_dict.get('value', [])
#             result['合集理由'] = clean_value(response_dict.get('note', []))
#         elif type == "感情状况":
#             result['感情状况'] = response_dict.get('value', [])
#             result['感情状况理由'] = clean_value(response_dict.get('note', []))
#         elif type == "最近的感觉":
#             result['最近的感觉'] = response_dict.get("最近的感觉", [])
#             result['什么让你有这种感觉'] = response_dict.get('什么让你有这种感觉', [])
#             result['最近的感觉理由'] = clean_value(response_dict.get('note', []))
        
#         logger.info(f"成功获取{type}标注")
#         return result
#     except Exception as e:
#         logger.error(f"获取{type}标注时发生错误: {e}")
#         raise
#         return {}

def get_annotation(row, type='合集'):
    role = f"noter-0205-{type}"
    role_prompt = get_role_prompt(role=role)
    
    # 根据不同类型生成提示词并返回相应的标注结果
    content = get_content_by_type(row, type)
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": content}
    ]
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,  # ✅ 更稳定的结构输出
            response_format={"type": "json_object"},
            n=1
        )

        response = completion.choices[0].message.content.strip()
        logger.warning(f"原始响应内容: {response}")  # ✅ 打印原始内容

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"出错内容如下：\n{response}")
            raise  # 或者 return {}

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
        
        logger.info(f"成功获取{type}标注")
        return result
    except Exception as e:
        logger.error(f"获取{type}标注时发生错误: {e}")
        # 可选择抛出错误或跳过本条数据
        # raise
        return {}  # ✅ 返回空字典，不中断主流程


def process_row(index, row, checkpoint_file, input_file, pbar):
    """处理每一行数据并保存进度"""
    logger.debug(f"正在处理第 {index} 行数据")  # 调试日志
    row_results = {}

    # 将自我肯定语添加为字典项
    row_results.update({'自我肯定语': row['自我肯定语']})
    
    for type in ['合集', '感情状况', '最近的感觉']:
        annotation_result = get_annotation(row, type)
        row_results.update(annotation_result)
    
    # 存储处理结果到文件
    save_results(row_results, input_file)

    # 更新checkpoint并记录日志
    update_checkpoint(checkpoint_file, index)
    logger.info(f"已更新checkpoint，已处理行：{index}")
    
    # 更新进度条
    Thread(target=update_progress, args=(pbar,)).start()

def load_checkpoint(checkpoint_file):
    """从文件加载checkpoint，返回已处理的行数"""
    checkpoint_data = set(get_checkpoint(checkpoint_file)) 
    return checkpoint_data

def process_file(input_file, k=None, from_retry=False):
    """主任务函数：处理文件中的每一行数据"""
    data = pd.read_csv(input_file)
    if k:
        data = data.sample(k)
    
    checkpoint_file = input_file.replace('.csv','_checkpoint.txt')
    output_file = input_file.replace('.csv','_result.csv')

    # 如果是从失败列表开始处理
    if from_retry:
        retry_output_file = input_file.replace('.csv', '_result_retry.csv')
        retry_data = pd.read_csv(retry_output_file)
        data = data[data['自我肯定语'].isin(retry_data['自我肯定语'])]

    completed_indices = load_checkpoint(checkpoint_file)
    print(f"从检查点文件读取到已完成的索引: {completed_indices}")
    
    with tqdm(total=len(data), initial=len(completed_indices), desc='进度', unit='task', position=0) as pbar:
        MAX_CONCURRENT_WORKERS = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
            futures = {}
            for index, row in data.iterrows():
                if index in completed_indices:
                    pbar.update(1)
                    continue
                future = executor.submit(
                    process_row, index, row, checkpoint_file, input_file, pbar
                )
                futures[future] = index
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                    Thread(target=update_progress, args=(pbar,)).start()
                    update_checkpoint(checkpoint_file, index)
                except Exception as e:
                    print(f"任务{index}失败: {e}")
                    raise

def main():
    input_file = '../data/0401句子更新 - Sheet1.csv'
    
    print("选择处理方式：")
    print("1. 从头开始")
    print("2. 从失败的列表开始")
    choice = input("请输入选项(1或2): ").strip()
    
    if choice == "1":
        process_file(input_file,k=500)
        # process_file(input_file)
    elif choice == "2":
        process_file(input_file, from_retry=True)
    
    # 显示进度
    retry_output_file = input_file.replace('.csv', '_result_retry.csv')
    output_file = input_file.replace('.csv', '_result.csv')
    
    completed = pd.read_csv(output_file)
    failed = pd.read_csv(retry_output_file)
    
    total_samples = len(completed) + len(failed)
    completed_samples = len(completed)
    failed_samples = len(failed)
    
    print(f"当前一共 {total_samples} 个样本，已完成 {completed_samples} 个，失败 {failed_samples} 个。")
    
    retry = input("是否立即重试从失败列表开始处理？(y/n): ").strip().lower()
    if retry == "y":
        process_file(input_file, from_retry=True)
    else:
        print("程序已退出。")
        sys.exit(0)

if __name__ == "__main__":
    main()
# retry有bug