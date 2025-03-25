import pandas as pd
import sys
import logging
import coloredlogs
from tqdm import tqdm
import concurrent.futures
import os
import json
from threading import Lock,Thread

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
logger.addHandler(console_handler)
coloredlogs.install(level='INFO', logger=logger)

checkpoint_lock = Lock()

def get_content_by_type(row, type):
    if type == '合集':
        message = f"""
        请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定标签的用户，并根据适配的标签返回JSON格式的结果。

        自我肯定语: {row['自我肯定语']}
        

        背景信息：
        自我肯定语的能量级别反映了其情感强度和目标导向。根据能量级别，可以将自我肯定语分为以下两类：
        1. **低能量级别**：
        - 通常涉及情感疗愈、心理修复或平和心境，适合需要安抚或支持的用户。
        - 示例标签：情感疗愈、治愈之旅、平和心境。
        2. **高能量级别**：
        - 通常具有强烈的激励性、目标导向或自我提升的性质，适合追求成长或积极行动的用户。
        - 示例标签：自信、个人成长、成为英雄。
        - **注意**：高能量级别的自我肯定语（如“我是最棒的”）不得标注为“情感疗愈”或“治愈之旅”，因为这些标签更适合低能量级别的情感支持场景。
        
        能量层级定义与标准
        L1（低能量级）
        定义：生存防护层，主要应对危机，提供即时安全感，维持心理存续。
        示例：
        情绪应对式：简单-情绪应对式
        描述：L1生存防护层——应对危机，提供即时安全感，维持心理存续。
        标注标准：句子或描述主要关注即时的安全感和危机应对。
        L2（低能量级）
        定义：稳态修复层，目标为修复创伤，重建内在平衡，培养积极心态。
        示例：
        安抚接纳式：简单-自我接纳式
        描述：L2稳态修复层，目标为修复创伤，重建内在平衡，培养积极心态。
        标注标准：句子或描述聚焦于修复和平衡，强调创伤后的恢复。
        L3（中能量级）
        定义：潜能认知层，目标为承认自身优点，提升自信，校正认知。
        示例：
        主体自信式：简单-主体自信式
        描述：L3潜能认知层，目标为承认自身优点，提升自信，校正认知。
        标注标准：句子或描述关注自我认知的提升和潜能的确认。
        L4（中能量级）
        定义：行动实现层，通过具体行动实现目标，重塑认知。
        示例：
        心念成长式：简单-心念培育式
        描述：L4行动实现层，通过微小、持续的心理操作，将抽象成长目标转化为可执行的行动，通过持续行为重塑认知。
        标注标准：句子或描述强调通过行动实现目标，注重行为的持续性和可执行性。
        L5（高能量级）
        定义：意义构建层，目标为构建价值和意义。
        示例：
        意义构建式：简单-主体意义式
        描述：L5意义构建层，目标为构建价值和意义。
        标注标准：句子或描述聚焦于价值和意义的构建，强调目标的深远性。
        L6（高能量级）
        定义：超越整合层，目标为超越自我，与更大的场域连接。
        示例：
        主权宣告式：简单-主权宣告式
        描述：L6超越整合层，目标为超越自我，与更大的场域连接。
        标注标准：句子或描述强调超越自我，连接更高层次的价值或目标。

        任务要求：
        1. 判断这个自我肯定语是否适合推送给以下标签的用户：
        - 情感疗愈
        - 自信
        - 平和心境
        - 治愈之旅
        - 人际交往
        - 自我关怀
        - 个人成长
        - 成为英雄
        2. 选择一个或多个标签进行标注，**不要**捏造标签，确保标签与自我肯定语的情感和语境相符。
        3. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。
        4. **特别注意**：对于能量级别很高的自我肯定语（如“我是最棒的”），不得标注为“情感疗愈”或“治愈之旅”。

        返回结果的格式要求：
        - `"key"`：固定值 `"合集"`。
        - `"value"`：一个列表，包含选择的标签。
        - `"note"`：一个字符串，说明未选择某些标签的理由。

        返回结果的示例格式：
        ```json
        {{
            "key": "合集",
            "value": ["自信", "个人成长"],
            "note": "未选择'情感疗愈'和'治愈之旅'，因为该自我肯定语能量级别较高，不适合情感疗愈场景；未选择'平和心境'，因为语句更倾向于激励而非平静。"
        }}
        """
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
    valid_collections = {"情感疗愈", "自信", "平和心境", "治愈之旅", "人际交往", "自我关怀", "个人成长", "成为英雄"}
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
    row_results_df.to_csv(save_file, mode='a', header=not os.path.exists(output_file), index=False)

def process_row(index, row, checkpoint_file, input_file, pbar):
    """处理每一行数据并保存进度"""
    logger.debug(f"正在处理第 {index} 行数据")  # 调试日志
    row_results = {}

    # 将自我肯定语添加为字典项
    row_results.update({'自我肯定语': row['自我肯定语']})
    
    for type in ['合集', '感情状况', '最近的感觉']:
        annotation_result = get_annotation(row, type)
        row_results.update(annotation_result)
    
    # 更新其它列
    # row_results.update({'场景': row['场景']})
    # row_results.update({'子场景': row['子场景']})
    # row_results.update({'用户需求': row['用户需求']})
    # row_results.update({'心理作用机制与功能': row['心理作用机制与功能']})
    # row_results.update({'句子级别': row['句子级别']})

    # 存储处理结果到文件
    save_results(row_results, input_file)

    # 更新checkpoint并记录日志
    update_checkpoint(checkpoint_file, index)
    logger.info(f"已更新checkpoint，已处理行：{index}")
    
    # 更新进度条
    Thread(target=update_progress, args=(pbar,)).start()


def clean_value(value):
    """清理字段中的换行符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ')  # 删除换行符和回车符
    return value


def load_checkpoint(checkpoint_file):
    """从文件加载checkpoint，返回已处理的行数"""
    checkpoint_data = set(get_checkpoint(checkpoint_file)) 
    return checkpoint_data

def process_file(input_file):
    """主任务函数：处理文件中的每一行数据"""
    data = pd.read_csv(input_file)
    # data = data.sample(10)
    checkpoint_file = input_file.replace('.csv','_checkpoint.txt')
    output_file = input_file.replace('.csv','_result.csv')
    
    # 2025-02-06 工作：
    # 检查output_file是否存在，checkpoint是否存在
    # 如果output_file存在但checkpoint不存在，就
    # 比对 data与 output_file的自我肯定语,生产者,场景,子场景,用户需求,心理作用机制与功能,句子级别
    # 如果一模一样，就把 i 添加到初始化的checkpoint.txt中
    
    completed_indices = load_checkpoint(checkpoint_file)
    print(f"从检查点文件读取到已完成的索引: {completed_indices}")
    
    # 使用进度条显示处理进度
    with tqdm(total=len(data),initial=len(completed_indices),desc='进度',unit='task',position=0) as pbar:
        MAX_CONCURRENT_WORKERS = 10
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
                    Thread(target=update_progress,args=(pbar,)).start()
                    update_checkpoint(checkpoint_file, index)
                except Exception as e:
                    print(f"任务{index}失败: {e}")
                    raise


# def get_annotation(row, type='合集'):
#     role = f"noter-0205-{type}"
#     role_prompt = get_role_prompt(role)
#     context = get_content_by_type(row, type)
#     user_prompt = f"{role_prompt}\n{context}"
#     result = client.ask(MODEL_NAME, user_prompt)
#     result_json = json.loads(result['data']['text'])
#     return result_json

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
            temperature=1,
            response_format={"type": "json_object"},  # 确保返回 JSON 格式
            n=1  # 请求返回1个结果
        )
        response = completion.choices[0].message.content.strip()
        response_dict = json.loads(response)
        
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
        raise
        return {}

if __name__ == "__main__":
    input_file = '../data/0315句子更新 - 汇总表_result_retry.csv'
    # input_file = '../data/test.csv'
    # checkpoint_file = input_file.replace('.csv','_checkpoint.txt')
    # process_file(input_file, checkpoint_file)
    process_file(input_file)


# import pandas as pd
# import sys
# import logging
# import coloredlogs  # 引入coloredlogs库
# from tqdm import tqdm  # 导入tqdm
# sys.path.append('../src')
# from prompt import get_role_prompt
# from kimi_api import client, MODEL_NAME
# import json

# # 配置日志
# logger = logging.getLogger()
# coloredlogs.install(level='INFO', logger=logger)

# def get_content_by_type(row, type):
#     if type == '合集':
#         message = f"""
#         请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定标签的用户，并根据适配的标签返回JSON格式的结果。

#         自我肯定语: {row['自我肯定语']}
#         场景: {row['场景']}
#         子场景: {row['子场景']}
#         用户需求: {row['用户需求']}
#         心理作用机制与功能: {row['心理作用机制与功能']}
#         句子级别: {row['句子级别']}
        
#         1. 判断这个自我肯定语是否适合推送给以下标签的用户：["情感疗愈", "自信", "平和心境", "治愈之旅", "人际交往", "自我关怀", "个人成长", "成为英雄"]。
#         2. 选择一个或多个标签进行标注，**不要**捏造标签，确保标签与自我肯定语的情感和语境相符。
#         3. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

        
#         请返回符合条件的标签列表，并以标准JSON格式返回结果，如下所示：
#         {{
#             "key": "合集",
#             "value": ["情感疗愈", "自信", "治愈之旅"],
#             "note":"不放一些标签的理由"
#         }}
#         """
#     elif type == '感情状况':
#         message = f"""
#         根据以下自我肯定语及其相关信息，判断是否适合推送给处于特定感情状况的用户，并根据适合的感情状况返回JSON格式的结果。

#         自我肯定语: {row['自我肯定语']}
#         场景: {row['场景']}
#         子场景: {row['子场景']}
#         用户需求: {row['用户需求']}
#         心理作用机制与功能: {row['心理作用机制与功能']}
#         句子级别: {row['句子级别']}

#         1. 判断该自我肯定语及其场景描述是否适合以下感情状况的用户：
#         ["正在恋爱", "最近刚分手", "处在一段艰难的亲密关系中", "快乐地单身着", "单身但准备好了开始新的恋情", "有点辛苦的暗恋"]。
#         2. 选择一个或多个感情状况标签进行标注，**不要**选择其他标签。
#         3. 返回的标签列表应该准确反映与该自我肯定语的适配情况。
#         4. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

#         请以如下格式返回结果：
#         {{
#             "key": "感情状况",
#             "value": ["正在恋爱", "单身但准备好了开始新的恋情"],
#             "note": "不放一些标签的理由"
#         }}
#         """
#     elif type == '最近的感觉':
#         message = f"""
#         请根据以下自我肯定语及其相关信息，判断它们是否适合推送给具有特定情感状态的用户，并根据用户的情感状态返回相应的JSON格式结果。

#         自我肯定语: {row['自我肯定语']}
#         场景: {row['场景']}
#         子场景: {row['子场景']}
#         用户需求: {row['用户需求']}
#         心理作用机制与功能: {row['心理作用机制与功能']}
#         句子级别: {row['句子级别']}

#         1. 判断该自我肯定语及其场景描述是否适合推送给以下情感状态的用户：
#         ["开心", "很好", "一般", "不好", "糟糕"]。
#         2. 请根据句子的情感倾向选择一个或多个标签：  
#         - 【开心】和【很好】可以同时出现，  
#         - 【不好】和【糟糕】也可以同时出现，  
#         - 不要选择额外的标签。
#         3. 1级句子倾向于【不好】和【糟糕】，2级句子倾向于【开心】和【很好】，但不是绝对的，具体分析。
#         4. "什么让你有这种感觉"是对"最近的感觉"的解释，请从以下选项中选择与感受相关的因素：
#         ["家庭", "朋友", "工作", "健康", "感情", "学业", "自己"]。
#         5. 如果不选择某些标签，则在"note"中给出理由。选择的标签不用给出。

#         请以如下格式返回结果：
#         {{
#             "最近的感觉": ["不好", "糟糕"],
#             "什么让你有这种感觉": ["家庭", "健康"],
#             "note": "不放一些标签的理由"
#         }}
#         """
#     return message

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
#             result['合集理由'] = response_dict.get('note', [])
#         elif type == "感情状况":
#             result['感情状况'] = response_dict.get('value', [])
#             result['感情状况理由'] = response_dict.get('note', [])
#         elif type == "最近的感觉":
#             result['最近的感觉'] = response_dict.get("最近的感觉", [])
#             result['什么让你有这种感觉'] = response_dict.get('什么让你有这种感觉', [])
#             result['最近的感觉理由'] = response_dict.get('note', [])
        
#         logger.info(f"成功获取{type}标注")
#         return result
#     except Exception as e:
#         logger.error(f"获取{type}标注时发生错误: {e}")
#         return {}

# def main():
#     input_path = "../data/0205标注协作表 - 上线句子清单0205.csv"
#     logger.info(f"开始加载数据集: {input_path}")
#     df = pd.read_csv(input_path)
#     df = df[['自我肯定语', '场景', '子场景', '用户需求', '心理作用机制与功能', '句子级别']]
#     # df = df.sample(1)
#     # 结果存储
#     results = []

#     # 使用tqdm包裹迭代器显示进度条
#     for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理数据"):
#         row_results = {}

#         # 将自我肯定语添加为字典项
#         row_results.update({'自我肯定语': row['自我肯定语']})
        
#         for type in ['合集', '感情状况', '最近的感觉']:
#             annotation_result = get_annotation(row, type)
#             row_results.update(annotation_result)
        
#         # 更新其它列
#         row_results.update({'场景': row['场景']})
#         row_results.update({'子场景': row['子场景']})
#         row_results.update({'用户需求': row['用户需求']})
#         row_results.update({'心理作用机制与功能': row['心理作用机制与功能']})
#         row_results.update({'句子级别': row['句子级别']})
        
#         results.append(row_results)

#     # 将结果保存到一个新的DataFrame并导出
#     result_df = pd.DataFrame(results)
#     print(result_df.columns)
#     result_df = result_df[["自我肯定语","合集","感情状况","最近的感觉","什么让你有这种感觉","合集理由","感情状况理由","最近的感觉理由","场景","子场景","用户需求","心理作用机制与功能","句子级别"]]
#     result_df.to_csv(input_path.replace('.csv', '_result.csv'), index=False)
#     logger.info(f"标注完成，结果已导出为{input_path.replace('.csv', '_result.csv')}")

# if __name__ == "__main__":
#     main()